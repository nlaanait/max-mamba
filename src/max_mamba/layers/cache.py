import numpy as np
from max.dtype import DType
from max.graph import TensorValue, Weight, ops

from max_mamba import Mamba2Config


class Mamba2Cache:
    """
    Arguments:
        config: Mamba2Config
        batch_size: int

    Attributes:
        dtype: (`DType`):
            The default `dtype` used to initializing the cache.
        conv_kernel_size: (`int`):
            Model's convolution kernel size taken from config.
        n_groups: (`int`):
            Model's number of groups taken from the config - similar to tensor parallel in Transformer.
        state_size: (`int`):
            Model's SSM state size taken from config.
        num_heads: (`int`):
            The number of heads used in the linear attention / SSM.
        head_dim: (`int`):
            The respective dimension of the heads used in the linear attention / SSM.
        intermediate_size: (`int`):
            Model's intermediate_size based on (expand * hidden_dim) from config.
        conv_states: (`TensorValue`):
            A tensor of shape `[num_layers, batch_size, conv_kernel_size, intermediate_size + 2 * n_groups * state_size]` that holds convolutional states.
        ssm_states: (`TensorValue`):
            A tensor of shape `[num_layers, batch_size, num_heads, head_dim, state_size]` that holds ssm states.
    """

    def __init__(
        self,
        config: Mamba2Config,
        batch_size: int,
    ):
        self.dtype = Mamba2Config.dtype
        self.conv_kernel_size = config.conv_kernel
        self.n_groups = config.n_groups
        self.state_size = config.state_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.intermediate_size = int(config.expand * config.hidden_size)
        self.conv_states = Weight(
            "conv_states",
            self.dtype,
            (
                config.num_hidden_layers,
                batch_size,
                self.intermediate_size + 2 * self.n_groups * self.state_size,
                self.conv_kernel_size,
            ),
        )
        self.ssm_states = Weight(
            "ssm_states",
            self.dtype,
            (
                config.num_hidden_layers,
                batch_size,
                self.num_heads,
                self.head_dim,
                self.state_size,
            ),
        )
        self.conv_mask = np.zeros(
            (
                config.num_hidden_layers,
                batch_size,
                self.intermediate_size + 2 * self.n_groups * self.state_size,
                self.conv_kernel_size,
            ),
            dtype=np.int8,
        )
        self.ssm_mask = np.zeros(
            (
                config.num_hidden_layers,
                batch_size,
                self.num_heads,
                self.head_dim,
                self.state_size,
            ),
            dtype=np.int8,
        )

    def update_conv_state(
        self, layer_idx: int, new_conv_state: TensorValue, cache_init: bool = False
    ) -> TensorValue:
        self.conv_mask[layer_idx, ...] = 1
        self.conv_states = ops.cast(self.conv_states, new_conv_state.dtype)
        if new_conv_state.device:
            self.conv_states: TensorValue = self.conv_states.to(new_conv_state.device)
        if cache_init:
            mask = ops.constant(self.conv_mask, DType.bool)
            self.conv_states = ops.masked_scatter(
                self.conv_states, mask, new_conv_state
            )
        else:
            # NEED To implement tensor roll
            # self.conv_states[layer_idx][:, :, -1] = new_conv_state[:, 0, :]
            return new_conv_state
        self.conv_mask[layer_idx, ...] = 0
        return self.conv_states[layer_idx]

    def update_ssm_state(
        self, layer_idx: int, new_ssm_state: TensorValue
    ) -> TensorValue:
        self.ssm_mask[layer_idx, ...] = 1
        self.ssm_states = ops.cast(self.ssm_states, new_ssm_state.dtype)
        if new_ssm_state.device:
            self.ssm_states: TensorValue = self.ssm_states.to(new_ssm_state.device)
        mask = ops.constant(self.ssm_mask, DType.bool)
        self.ssm_states = ops.masked_scatter(self.ssm_states, mask, new_ssm_state)
        self.ssm_mask[layer_idx, ...] = 0
        return self.ssm_states[layer_idx]

    def reset(self):
        self.conv_states = ops.mul(self.conv_states, 0)
        self.ssm_states = ops.mul(self.ssm_states, 0)
