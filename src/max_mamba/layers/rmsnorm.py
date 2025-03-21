from dataclasses import dataclass
from typing import Optional
import numpy as np
from max.dtype import DType
from max.graph import (
    TensorValue,
    ops,
)

from max.nn import Module


class RMSNormGated(Module):
    def __init__(
        self,
        hidden_size: tuple[int, ...],
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps

    def __call__(  # type: ignore
        self, h: TensorValue, gate: Optional[TensorValue] = None
    ) -> TensorValue:
        self.weight = ops.constant(np.ones_like(h), dtype=DType.float32)
        weight: TensorValue = ops.cast(self.weight, h.dtype)
        if h.device:
            weight = weight.to(h.device)
        if gate is not None:
            gate = ops.cast(gate, h.dtype)
            if h.device:
                gate = gate.to(h.device)
            h = ops.mul(h, ops.silu(gate))
        variance = ops.mean(ops.pow(h, 2), axis=-1)
        h = ops.mul(h, ops.rsqrt(ops.add(variance, self.eps)))
        return h
