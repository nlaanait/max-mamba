from typing import Optional

from max import nn
from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops

from max_mamba.config import Mamba2Config
from max_mamba.layers.cache import Mamba2Cache
from max_mamba.layers.mixer import Mamba2Mixer
from max_mamba.layers.rmsnorm import Mamba2RMSNorm


class Mamba2Block(nn.Module):
    def __init__(
        self,
        config: Mamba2Config,
        layer_idx: int,
        device: DeviceRef = DeviceRef.CPU(),
        dtype: DType = DType.float32,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = Mamba2RMSNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )
        self.mixer = Mamba2Mixer(
            config=config, layer_idx=layer_idx, device=device, dtype=dtype
        )

    def __call__(
        self,
        hidden_states,
        cache_params: Optional[Mamba2Cache] = None,
        cache_position: Optional[list[int]] = None,
        attention_mask: Optional[TensorValue] = None,
    ):
        residual = hidden_states
        hidden_states = self.norm(ops.cast(hidden_states, self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = ops.cast(residual, DType.float32)

        hidden_states = self.mixer(
            hidden_states=hidden_states,
            cache_params=cache_params,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )

        hidden_states = residual + hidden_states
        return hidden_states
