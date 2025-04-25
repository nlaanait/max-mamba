from typing import Optional

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.nn import Module


class RMSNormGated(Module):
    def __init__(
        self,
        hidden_size: tuple[int, ...],
        eps: float = 1e-6,
        dtype: DType | None = None,
        device: DeviceRef | None = None,
        name: str | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.device = device if device else DeviceRef.CPU()
        self.dtype = dtype if dtype else DType.float32
        self.weight = Weight(
            f"{name}.Weight" if name else "Weight",
            self.dtype,
            self.hidden_size,
            device=self.device,
        )

    def __call__(  # type: ignore
        self, h: TensorValue, gate: Optional[TensorValue] = None
    ) -> TensorValue:
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
