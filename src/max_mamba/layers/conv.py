from max import nn
from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight


class Conv1d(nn.Module):
    weight: Weight
    bias: Weight | None = None
    device: DeviceRef | None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        groups: int = 1,
        stride: int = 1,
        has_bias: bool = True,
        name: str | None = None,
        device: DeviceRef | None = None,
        dtype: DType | None = None,
    ):
        super().__init__()
        self.device = device if device else DeviceRef.CPU()
        self.dtype = dtype if dtype else DType.float32
        weight_shape = (
            kernel_size,
            int(in_channels / groups),
            out_channels,
        )
        self.weight = Weight(
            name=f"{name}.weight" if name else "weight",
            dtype=self.dtype,
            shape=weight_shape,
            device=self.device,
        )

        if has_bias:
            bias_shape = (out_channels,)
            self.bias = Weight(
                name=f"{name}.bias" if name else "bias",
                dtype=self.dtype,
                shape=bias_shape,
                device=self.device,
            )
        else:
            self.bias = None
        self.conv = nn.Conv1D(
            filter=self.weight,
            bias=self.bias,
            padding=padding,
            groups=groups,
            stride=stride,
        )

    def __call__(self, x: TensorValue) -> TensorValue:  # type: ignore
        weight: TensorValue = self.weight
        if self.device:
            weight = weight.to(self.device)

        return self.conv(x)
