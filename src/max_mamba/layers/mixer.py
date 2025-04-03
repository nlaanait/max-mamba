from typing import Optional
from max_mamba import Mamba2Config
from max_mamba.layers import RMSNormGated, Conv1d, Mamba2Cache
from max import nn
from max.graph import ops, Weight, TensorValue
import numpy as np


def apply_mask_to_padding_states(
    hidden_states: TensorValue, attention_mask: TensorValue | None
) -> TensorValue:
    if attention_mask is not None:
        attention_mask = ops.cast(attention_mask, hidden_states.dtype)
        attention_mask = ops.unsqueeze(attention_mask, axis=-1)
        attention_mask = ops.broadcast_to(
            attention_mask,
            (hidden_states.shape[0], hidden_states.shape[1], hidden_states.shape[2]),
        )
        hidden_states = ops.mul(hidden_states, attention_mask)

    return hidden_states


def pad_tensor(
    x: TensorValue, pad: tuple, value=0, mode: str = "constant"
) -> TensorValue:
    if mode != "constant":
        raise NotImplementedError("mode != 'constant' not implemented.")

    # store pad dimensions, starting from last axis
    pad_dim = {
        -1 - idx // 2: [pad[idx], pad[idx + 1]] for idx in range(0, len(pad) - 1, 2)
    }

    for dim, val in pad_dim.items():
        if sum(val) < 1:
            continue

        # compute shapes
        pad_shape_left = list(x.shape)
        pad_shape_right = list(x.shape)
        pad_shape_left[dim] = val[0]
        pad_shape_right[dim] = val[1]
        pad_shape_left = [int(itm) for itm in pad_shape_left]
        pad_shape_right = [int(itm) for itm in pad_shape_right]

        # Create padding arrays
        pad_left = ops.constant(
            value=np.full(shape=pad_shape_left, fill_value=value), dtype=x.dtype
        )
        pad_right = ops.constant(
            value=np.full(shape=pad_shape_right, fill_value=value), dtype=x.dtype
        )
        x = ops.concat([pad_left, x, pad_right], axis=dim)
    return x


class Mamba2Mixer(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(self, config: Mamba2Config, layer_idx: int):
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = int(config.expand * self.hidden_size)
        self.time_step_rank = int(config.time_step_rank)
        self.layer_idx = layer_idx
        self.use_conv_bias = config.use_conv_bias
        self.activation = config.hidden_act
        self.act = ops.silu

        self.layer_norm_epsilon = config.layer_norm_epsilon
        self.rms_norm = config.rms_norm

        self.n_groups = config.n_groups
        self.head_dim = config.head_dim
        self.chunk_size = config.chunk_size

        self.time_step_limit = config.time_step_limit
        self.time_step_min = config.time_step_min
        self.time_step_max = config.time_step_max

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        self.conv1d = Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            has_bias=self.use_conv_bias,
            kernel_size=self.conv_kernel_size,
            groups=self.n_groups,
            padding=config.conv_kernel - 1,
            dtype=config.dtype,
        )

        # projection of the input hidden states
        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.LinearV2(
            self.hidden_size,
            projection_size,
            dtype=config.dtype,
            has_bias=config.use_bias,
        )
        # selective projection used to make dt, B and C input dependant

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = ops.constant(np.ones(self.num_heads), dtype=config.dtype)

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = ops.constant(np.arange(1, self.num_heads), dtype=config.dtype)
        self.A_log = ops.log(A)
        self.norm = RMSNormGated(
            (self.intermediate_size,), eps=config.layer_norm_epsilon
        )
        self.D = ops.constant(np.ones(self.num_heads), dtype=config.dtype)

        self.out_proj = nn.LinearV2(
            self.intermediate_size,
            self.hidden_size,
            dtype=config.dtype,
            has_bias=config.use_bias,
        )
        self.use_bias = config.use_bias

    def __call__(
        self,
        input_states: TensorValue,
        cache_params: Optional[Mamba2Cache] = None,
        cache_position: Optional[int] = None,
        attention_mask: Optional[TensorValue] = None,
    ):
        # This implements torch_forward in Transformer.Mamba2Mixer
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype

        # 1. Gated MLP's linear projection
        input_states = apply_mask_to_padding_states(input_states, attention_mask)
        projected_states = self.in_proj(input_states)
        d_mlp = (
            projected_states.shape[-1]
            - 2 * self.intermediate_size
            - 2 * self.n_groups * self.ssm_state_size
            - self.num_heads
        ) // 2
        _, _, gate, hidden_states_B_C, dt = ops.split(
            projected_states,
            [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads],
            axis=-1,
        )

        # 2. Convolution Sequence Transformation
        if (
            cache_params is not None
            and cache_position is not None
            and cache_position > 0
        ):
            cache_params.update_conv_state(
                layer_idx=self.layer_idx,
                new_conv_state=hidden_states_B_C,
                cache_init=False,
            )

            # TODO: check if device assignment is necessary since it's already in the cache
            conv_states = cache_params.conv_states[self.layer_idx].to(
                device=input_states.device.CPU() if input_states.device else None,
            )

            hidden_states_B_C = ops.sum(
                conv_states * ops.squeeze(self.conv1d.weight, axis=1), axis=-1
            )
            if self.use_conv_bias:
                hidden_states_B_C = ops.add(hidden_states_B_C, self.conv1d.bias)
            hidden_states_B_C = self.act(hidden_states_B_C)
        else:
            # Initialize Cache
            if cache_params is not None:
                hidden_states_B_C_T = hidden_states_B_C.transpose(1, 2)
                conv_states = pad_tensor(
                    hidden_states_B_C_T,
                    (cache_params.conv_kernel_size - hidden_states_B_C_T.shape[-1], 0),
                )
                cache_params.update_conv_state(
                    layer_idx=self.layer_idx,
                    new_conv_state=conv_states,
                    cache_init=True,
                )
            hidden_states_B_C = self.act(
                self.conv1d(hidden_states_B_C.transpose(1, 2))[..., :seq_len].transpose(
                    1, 2
                )
            )
        hidden_states_B_C = apply_mask_to_padding_states(
            hidden_states_B_C, attention_mask
        )
        hidden_states, B, C = ops.split(
            hidden_states_B_C,
            [
                self.intermediate_size,
                self.n_groups * self.ssm_state_size,
                self.n_groups * self.ssm_state_size,
            ],
            axis=-1,
        )

        # 3. SSM Transformation
        return hidden_states
