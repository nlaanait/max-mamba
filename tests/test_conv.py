from max.engine.api import InferenceSession
from max.dtype import DType
from max.graph import Graph, TensorType
from max_mamba.layers.conv import Conv1d
import numpy as np

import torch
import torch.nn as nn

torch.manual_seed(1234)
np.random.seed(1234)


def init_input_tensor(size: tuple[int, ...]) -> np.ndarray:
    # return np.random.rand(*size).astype(np.float32)
    return np.ones(size, dtype=np.float32)


def init_w_and_b_tensors(
    kernel_shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    weight = np.random.randn(*kernel_shape).astype(np.float32)
    bias = np.random.rand(kernel_shape[2]).astype(np.float32)
    return weight, bias


def init_w_and_b_tensors_kaiming(
    kernel_shape: tuple[int, ...], groups: int, in_channels: int
) -> tuple[np.ndarray, np.ndarray]:
    weight = np.sqrt(groups / (in_channels * kernel_shape[0])) * np.random.uniform(
        low=-1,
        high=1,
        size=(kernel_shape[0], int(in_channels / groups), kernel_shape[2]),
    ).astype(np.float32)
    bias = np.sqrt(groups / (in_channels * kernel_shape[0])) * np.random.uniform(
        low=-1, high=1, size=(kernel_shape[2],)
    ).astype(np.float32)
    return weight, bias


def conv1d_test():
    in_channels = 4
    out_channels = 6
    kernel_size = 2
    batch_size = 1
    seq_length = 10
    padding = 0
    groups = 1
    has_bias = False

    # Initialize input and weight tensors
    input_size = (batch_size, seq_length, in_channels)
    input_tensor = init_input_tensor(size=input_size)
    weight, bias = init_w_and_b_tensors(
        kernel_shape=(kernel_size, in_channels, out_channels)
    )

    # PyTorch Conv1d
    pt_conv1d = nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=padding,
        groups=groups,
        bias=has_bias,
    )
    pt_conv1d.weight = nn.Parameter(torch.from_numpy(weight).permute(2, 1, 0))
    if has_bias:
        pt_conv1d.bias = nn.Parameter(torch.from_numpy(bias))
    pt_input = torch.from_numpy(input_tensor).permute(0, 2, 1)
    pt_output = pt_conv1d(pt_input).permute(0, 2, 1).detach().numpy()

    # max_mamba Conv1d
    max_output = max_conv1d(
        input_tensor=input_tensor,
        weight=weight,
        bias=bias,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=padding,
        groups=groups,
        has_bias=has_bias,
    )

    # Compare outputs
    assert np.allclose(pt_output, max_output, atol=1e-5)
    print("Passed Conv1d Test")


def max_conv1d(
    input_tensor: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None,
    has_bias: bool,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    padding: int,
    groups: int,
) -> np.ndarray:
    input_size = tuple(input_tensor.shape)
    max_conv1d = Graph(
        "conv1d",
        Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dtype=DType.float32,
            has_bias=has_bias,
            groups=groups,
        ),
        input_types=[TensorType(DType.float32, input_size)],
    )

    session = InferenceSession()
    model = session.load(
        max_conv1d,
        weights_registry={
            "weight": weight,
            "bias": bias,
        },  # type: ignore
    )
    result = model.execute(input_tensor)[0]
    return result.to_numpy()


if __name__ == "__main__":
    conv1d_test()
