import numpy as np
import pytest
import torch
import torch.nn as nn
from max.dtype import DType
from max.engine.api import InferenceSession
from max.graph import Graph, TensorType

from max_mamba.layers.conv import Conv1d

torch.manual_seed(1234)
np.random.seed(1234)


@pytest.fixture
def init_w_and_b_tensors():
    def _init(kernel_shape):
        weight = np.random.rand(*kernel_shape).astype(np.float32)  # type: ignore
        bias = np.random.rand(kernel_shape[-1]).astype(np.float32)
        return weight, bias

    return _init


@pytest.fixture
def init_np_tensor():
    def _init(size):
        return np.ones(size, dtype=np.float32)

    return _init


@pytest.fixture
def torch_seed():
    torch.manual_seed(1234)
    np.random.seed(1234)


def get_max_conv1d_result(
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


def test_conv1d(RTOL, init_np_tensor, init_w_and_b_tensors):
    # Test parameters
    in_channels = 4
    out_channels = 6
    kernel_size = 2
    batch_size = 1
    seq_length = 10
    padding = 0
    groups = 1
    has_bias = False

    # Initialize tensors
    input_size = (batch_size, seq_length, in_channels)
    input_tensor = init_np_tensor(size=input_size)
    weight, bias = init_w_and_b_tensors(
        kernel_shape=(kernel_size, in_channels, out_channels)
    )

    # PyTorch implementation
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

    # MAX implementation
    max_output = get_max_conv1d_result(
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

    np.testing.assert_allclose(pt_output, max_output, rtol=RTOL)
