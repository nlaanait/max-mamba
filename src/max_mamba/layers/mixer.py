from typing import Optional
from max_mamba import Mamba2Config
from max_mamba.layers import RMSNormGated, Conv1d, Mamba2Cache
from max import nn
from max.dtype import DType
from max.graph import ops, Weight, TensorValue

import numpy as np

from max_mamba.ops import pad_tensor, softplus, clamp_tensor


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
        self.dtype = config.dtype

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
        # This ports torch_forward in Transformer.Mamba2Mixer
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
        A = -ops.exp(self.A_log)  # [num_heads]
        if (
            cache_params is not None
            and cache_position is not None
            and cache_position > 0
        ):
            cache_device = cache_params.ssm_states.device

            # Note: there is no need to pad parameter matrices here, as there is just one new token
            # for batched generation
            dt = dt[:, 0, :][:, None, ...]
            dt = dt.transpose(1, 2).broadcast_to(
                (batch_size, list(dt.shape)[-1], self.head_dim)
            )
            # [num_heads] -> [num_heads, head_dim]
            dt_bias = self.dt_bias[..., None].broadcast_to(
                (self.dt_bias.shape[0], self.head_dim)
            )

            dt = softplus(dt + dt_bias)
            dt = clamp_tensor(dt, self.time_step_limit[0], self.time_step_limit[1])
            A = A[..., None, None].broadcast_to(
                (self.num_heads, self.head_dim, self.ssm_state_size)
            )
            # [bsz, num_heads, head_dim, state_size]
            dA = ops.exp(dt[..., None] * A)

            # Discretize B
            # [bsz, n_groups * state_size] -> [bsz, n_groups, 1, state_size] ->
            # -> [bsz, n_groups, group to head repetition factor, state_size] -> [bsz, num_heads, state_size]
            B = B.reshape((batch_size, self.n_groups, -1))[..., None, :]
            B = B.broadcast_to(
                (
                    batch_size,
                    self.n_groups,
                    self.num_heads // self.n_groups,
                    B.shape[-1],
                )
            )
            B = B.reshape((batch_size, -1, B.shape[-1]))
            # [bsz, num_heads, head_dim, state_size]
            dB = dt[..., None] * B[..., None, :]

            # Discretize x into dB
            # [bsz, intermediate_size] -> [bsz, num_heads, head_dim]
            hidden_states = hidden_states.reshape((batch_size, -1, self.head_dim))
            dBx = dB * hidden_states[..., None]

            # State calculation
            cache_params.update_ssm_state(
                layer_idx=self.layer_idx,
                new_ssm_state=cache_params.ssm_states[self.layer_idx] * dA + dBx,
            )

            # Subsequent output
            # [bsz, n_groups * state_size] -> [bsz, num_heads, state_size]
            C = C.reshape((batch_size, self.n_groups, -1))[..., None, :]
            C = C.broadcast_to(
                (
                    batch_size,
                    self.n_groups,
                    self.num_heads // self.n_groups,
                    C.shape[-1],
                )
            )
            C = C.reshape((batch_size, -1, C.shape[-1]))
            # [bsz, num_heads, head_dim]

            ssm_states = cache_params.ssm_states[self.layer_idx]  # Shape: [b, h, d, n]
            # Reshape ssm_states to merge the first two dimensions
            ssm_states_reshaped = ssm_states.reshape(
                (batch_size * self.num_heads, self.head_dim, self.ssm_state_size)
            )  # Shape: [b*h, d, n]
            C_reshaped = C.reshape(
                (batch_size * self.num_heads, self.ssm_state_size, 1)
            )  # Shape: [b*h, n, 1]
            y = ops.matmul(ssm_states_reshaped, C_reshaped)
            y = y.reshape((batch_size, self.num_heads, self.head_dim))

            # D skip connect
            # [num_heads] -> [num_heads, head_dim]
            D = self.D[..., None].broadcast_to((list(self.D.shape)[0], self.head_dim))
            y = y + hidden_states * D

            # [bsz, num_heads, head_dim] -> [bsz, 1, intermediate_size]
            y = y.reshape((batch_size, -1))[:, None, ...]
        else:
            # ssd implementation (can be done with einsums)
            dt = softplus(dt + self.dt_bias)
            dt = clamp_tensor(dt, self.time_step_limit[0], self.time_step_limit[1])
            hidden_states = ops.cast(
                hidden_states.reshape((batch_size, seq_len, -1, self.head_dim)),
                DType.float32,
            )
            B = ops.cast(
                B.reshape((batch_size, seq_len, -1, self.ssm_state_size)), DType.float32
            )
            C = ops.cast(
                C.reshape((batch_size, seq_len, -1, self.ssm_state_size)), DType.float32
            )
            
