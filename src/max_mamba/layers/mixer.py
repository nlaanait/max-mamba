import math
from pathlib import Path
from typing import Optional

import numpy as np
from max import nn
from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops

from max_mamba import Mamba2Config
from max_mamba.layers import Mamba2Cache, RMSNormGated
from max_mamba.ops import clamp_tensor, pad_tensor, softplus, tile_tensor

mojo_ops_path = Path(__file__).parent / "kernels"
kernels = [
    "selective_state_update.mojo",
    "mamba_chunk_scan_combined.mojo",
    "mamba_split_conv1d_scan_combined.mojo",
    "causal_conv1d_fn.mojo",
    "causal_conv1d_update.mojo",
]
CUSTOM_MOJO_OPS = all([(mojo_ops_path / kernel).exists() for kernel in kernels])


def pad_tensor_by_size(input_tensor: TensorValue, pad_size: int) -> TensorValue:
    """
    Padding x tensor with `pad_size` on the seq_len dim (dim=1)

    Assumes that we only have tensors of either size 4 or 3
    """
    pad_shape = (
        (0, 0, 0, 0, 0, pad_size, 0, 0)
        if len(input_tensor.shape) == 4
        else (0, 0, 0, pad_size, 0, 0)
    )

    return pad_tensor(input_tensor, pad_shape, mode="constant", value=0)


def reshape_into_chunks(
    input_tensor: TensorValue, pad_size: int, chunk_size: int
) -> TensorValue:
    """
    Padding input_tensor with `pad_size` on the seq_len dim (dim=1) and
    simultaneously splitting it into chunk sequences.

    Assumes that we only have tensors of either size 4 or 3
    """
    # [bsz, seq_len, ...] -> [bsz, seq_len multiple of chunk_size, ...]
    input_tensor = pad_tensor_by_size(input_tensor, pad_size)

    if len(input_tensor.shape) == 3:
        # [bsz, seq_len multiple of chunk_size, num_heads] -> [bsz, -1, chunk_size, num_heads]
        return input_tensor.reshape(
            (input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2])
        )
    else:
        # [bsz, seq_len multiple of chunk_size, num_heads, head_dim or state_size] -> [bsz, -1, chunk_size, num_heads, head_dim or state_size]
        return input_tensor.reshape(
            (
                input_tensor.shape[0],
                -1,
                chunk_size,
                input_tensor.shape[2],
                input_tensor.shape[3],
            )
        )


def segment_sum(input_tensor: TensorValue) -> TensorValue:
    """
    More stable segment sum calculation. Uses cumulative sums and masking instead of direct subtractions.
    """
    chunk_size = int(input_tensor.shape[-1])

    # 1. expand input tensor to have an additional dimension and repeat along that dimension
    # [..., chunk_size] -> [..., chunk_size, chunk_size]
    new_shape = tuple(input_tensor.shape) + (chunk_size,)
    input_tensor = input_tensor[..., None].broadcast_to(new_shape)
    # 2. create a lower triangular mask with the diagonal set to 0
    mask_ = ~np.tril(
        np.ones(
            (
                chunk_size,
                chunk_size,
            ),
            dtype=np.int8,
        ),
        -1,
    ).astype(np.bool_)

    mask = ops.constant(
        mask_,
        dtype=DType.bool,
        device=input_tensor.device,
    )
    zeros = ops.constant(
        np.zeros(tuple(int(s) for s in input_tensor.shape), dtype=np.float32),
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    input_tensor = ops.masked_scatter(input_tensor, mask, zeros)
    # 3. compute actual cumsum
    tensor_segsum = ops.cumsum(input_tensor, axis=-2)

    # 4. apply mask to keep only the lower triangular part of the cumsum result (include diag)
    mask_ = ~np.tril(
        np.ones(
            (
                chunk_size,
                chunk_size,
            ),
            dtype=np.int8,
        ),
    ).astype(np.bool_)
    mask = ops.constant(
        mask_,
        dtype=DType.bool,
        device=input_tensor.device,
    )
    infs = ops.constant(
        np.full(
            tuple(int(s) for s in input_tensor.shape), float("-inf"), dtype=np.float32
        ),
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    tensor_segsum = ops.masked_scatter(tensor_segsum, mask, infs)
    return tensor_segsum


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


def mamba2_mixer_initializer(config: Mamba2Config):
    dt = np.exp(
        np.random.rand(config.num_heads)
        * (math.log(config.time_step_max) - np.log(config.time_step_min))
        + math.log(config.time_step_min)
    ).clip(min=config.time_step_floor)

    inv_dt = dt + np.log(-np.expm1(-dt))
    return {
        "A": np.arange(1, config.num_heads + 1, dtype=np.float32),
        "dt_bias": inv_dt.astype(np.float32),
        "D": np.ones(config.num_heads, dtype=np.float32),
    }


def mamba2_mixer_random_initializer(config: Mamba2Config):
    out_channels = (
        int(config.expand * config.hidden_size)
        + 2 * config.n_groups * config.state_size
    )
    in_channels = out_channels // config.n_groups
    conv_kernel_shape = (config.conv_kernel, in_channels, out_channels)
    in_proj_weight_dims = (
        int(
            2 * config.expand * config.hidden_size
            + 2 * config.n_groups * config.state_size
            + config.num_heads
        ),
        config.hidden_size,
    )
    out_proj_weight_dims = (config.hidden_size, int(config.expand * config.hidden_size))
    return {
        "conv1d.weight": np.random.rand(*conv_kernel_shape).astype(np.float32),
        "conv1d.bias": (
            np.zeros(out_channels)
            if config.use_conv_bias
            else np.random.rand(out_channels).astype(np.float32)
        ),
        "in_proj.weight": np.random.rand(*in_proj_weight_dims).astype(np.float32),
        "in_proj.bias": np.zeros(in_proj_weight_dims[-1]).astype(np.float32),
        "out_proj.weight": np.random.rand(*out_proj_weight_dims).astype(np.float32),
        "out_proj.bias": np.zeros(out_proj_weight_dims[-1]).astype(np.float32),
        "norm.weight": np.random.rand(out_proj_weight_dims[-1]).astype(np.float32),
    }


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
        self.time_step_rank = config.time_step_rank
        self.layer_idx = layer_idx
        self.use_conv_bias = config.use_conv_bias
        self.activation = config.hidden_act
        self.act = ops.silu
        self.dtype = config.dtype
        self.device = config.devices[0] if config.devices else DeviceRef.CPU()

        self.layer_norm_epsilon = config.layer_norm_epsilon
        self.rms_norm = config.rms_norm

        self.n_groups = config.n_groups
        self.head_dim = config.head_dim
        self.chunk_size = config.chunk_size

        self.time_step_limit = config.time_step_limit
        self.time_step_min = config.time_step_min
        self.time_step_max = config.time_step_max

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        self.conv1d = nn.Conv1D(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            has_bias=self.use_conv_bias,
            kernel_size=self.conv_kernel_size,
            num_groups=self.n_groups,
            padding=config.conv_kernel - 1,
            dtype=config.dtype,
            name="conv1d",
        )

        # projection of the input hidden states
        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(
            self.hidden_size,
            projection_size,
            dtype=config.dtype,
            has_bias=config.use_bias,
            name="in_proj",
            device=self.device,
        )
        # selective projection used to make dt, B and C input dependant

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = Weight("dt_bias", config.dtype, (self.num_heads,), self.device)

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        self.A = Weight("A", config.dtype, (self.num_heads,), self.device)
        self.norm = RMSNormGated(
            (self.intermediate_size,), eps=config.layer_norm_epsilon, name="norm"
        )
        self.D = Weight("D", config.dtype, (self.num_heads,), self.device)

        self.out_proj = nn.Linear(
            self.intermediate_size,
            self.hidden_size,
            dtype=config.dtype,
            has_bias=config.use_bias,
            name="out_proj",
            device=self.device,
        )
        self.use_bias = config.use_bias

    def __call__(
        self,
        hidden_states: TensorValue,
        cache_params: Optional[Mamba2Cache] = None,
        cache_position: Optional[list[int]] = None,
        attention_mask: Optional[TensorValue] = None,
    ):
        if CUSTOM_MOJO_OPS:
            raise NotImplementedError
        dtype = hidden_states.dtype
        if (
            attention_mask is not None
            and int(attention_mask.shape[1]) > 1
            and int(attention_mask.shape[0]) > 1
        ):
            # tune out hidden states for pad tokens, see https://github.com/state-spaces/mamba/issues/66
            hidden_states = ops.cast(
                hidden_states * attention_mask[:, :, None], dtype=dtype
            )
        return self.max_forward(
            hidden_states, cache_params, cache_position, attention_mask
        )

    def max_forward(
        self,
        input_states: TensorValue,
        cache_params: Optional[Mamba2Cache] = None,
        cache_position: Optional[list[int]] = None,
        attention_mask: Optional[TensorValue] = None,
    ):
        self.A_log = ops.log(self.A)
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
            and cache_position[0] > 0
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
            hidden_states_B_C = self.act(self.conv1d(hidden_states_B_C)[:, :seq_len, :])
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
            and cache_position[0] > 0
        ):

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
            B = tile_tensor(B, dims=(1, 1, self.num_heads // self.n_groups, 1))
            C = tile_tensor(C, dims=(1, 1, self.num_heads // self.n_groups, 1))
            pad_size = (
                self.chunk_size - int(seq_len) % self.chunk_size
            ) % self.chunk_size

            D_residual = self.D[..., None] * pad_tensor_by_size(hidden_states, pad_size)

            # Discretize x and A
            hidden_states = hidden_states * dt[..., None]
            A = ops.cast(A, hidden_states.dtype) * dt

            # Rearrange into blocks/chunks
            hidden_states, A, B, C = [
                reshape_into_chunks(t, pad_size, self.chunk_size)
                for t in (hidden_states, A, B, C)
            ]

            # [bsz, -1, chunk_size, num_heads] -> [bsz, num_heads, -1, chunk_size]
            A = ops.permute(A, dims=[0, 3, 1, 2])
            A_cumsum = ops.cumsum(A, axis=-1)

            # 1. Compute the output for each intra-chunk (diagonal blocks)
            # This is the analog of a causal mask
            L = ops.exp(segment_sum(A))
            # Contraction of C and B to get G (attention-weights like)
            G_intermediate = (
                C[:, :, :, None, :, :] * B[:, :, None, :, :, :]
            )  # shape: (b, c, l, s, h, n)
            G = ops.squeeze(
                ops.sum(G_intermediate, axis=-1), axis=-1
            )  # shape: (b, c, l, s, h)

            # Compute M, equivalent to applying attention mask to weights
            M_intermediate = G[..., None] * L.permute(dims=[0, 2, 3, 4, 1])[..., None]
            M = ops.squeeze(ops.sum(M_intermediate, axis=-1), axis=-1)

            # Compute Y_diag (apply to values)
            Y_diag = ops.squeeze(
                ops.sum(M[..., None] * hidden_states[:, :, None], axis=3), axis=3
            )

            # 2. Compute the state for each intra-chunk
            # (right term of low-rank factorization of off-diagonal blocks; B terms)
            decay_states = ops.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
            B_decay = B * ops.permute(decay_states, dims=[0, -2, -1, 1])[..., None]
            states = ops.squeeze(
                ops.sum(B_decay[..., None, :] * hidden_states[..., None], axis=2),
                axis=2,
            )
            # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
            # (middle term of factorization of off-diag blocks; A terms)
            if (
                cache_params is not None
                and cache_position is not None
                and cache_position[0] > 0
            ):
                previous_states = cache_params.ssm_states[self.layer_idx][
                    :, None, ...
                ]  # TODO: check device assignments of cache data
            else:
                previous_states = ops.constant(
                    np.zeros(
                        tuple(int(s) for s in states[:, :1].shape),
                        dtype=np.float32,  # TODO: remove fixed dtype
                    ),
                    dtype=states.dtype,
                    device=states.device,
                )
            states = ops.concat([previous_states, states], axis=1)
            decay_chunk = ops.exp(
                segment_sum(pad_tensor(A_cumsum[:, :, :, -1], (1, 0)))
            )
            decay_chunk = decay_chunk.transpose(1, 3)
            new_states = ops.squeeze(
                ops.sum(
                    (decay_chunk[..., None, None] * states[:, :, None, ...]), axis=1
                ),
                axis=1,
            )
            states, ssm_state = new_states[:, :-1], new_states[:, -1]

            # 4. Compute state -> output conversion per chunk
            # (left term of low-rank factorization of off-diagonal blocks; C terms)
            state_decay_out = ops.exp(A_cumsum)
            C_times_states = C[..., None, :] * states[:, :, None, ...]
            state_decay_out_permuted = ops.permute(state_decay_out, dims=[0, 2, 3, 1])
            Y_off = (
                ops.squeeze(ops.sum(C_times_states, axis=-1), axis=-1)
                * state_decay_out_permuted[..., None]
            )
            # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
            y = Y_diag + Y_off
            # [bsz, -1, self.chunk_size, num_heads, head_dim] -> [bsz, (padded) seq_len, num_heads, head_dim]
            y = y.reshape((batch_size, -1, self.num_heads, self.head_dim))

            y = y + D_residual
            # Cutting off padded chunks
            if pad_size > 0:
                y = y[:, :seq_len, :, :]
            y = y.reshape((batch_size, seq_len, -1))

            # Initialize Cache
            if ssm_state is not None and cache_params is not None:
                cache_params.update_ssm_state(
                    layer_idx=self.layer_idx, new_ssm_state=ssm_state
                )

        scan_output = self.norm(y, gate)

        # end ssd naive

        # 4. Final linear projection
        contextualized_states = self.out_proj(
            scan_output
        )  # [batch, seq_len, hidden_size]

        return contextualized_states
