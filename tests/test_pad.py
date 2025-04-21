import numpy as np
import pytest
import torch
import torch.nn.functional as F
from max.dtype import DType
from max.engine.api import InferenceSession
from max.graph import Graph, TensorType

from max_mamba.ops import pad_tensor


def get_max_pad_result(
    input_tensor: np.ndarray,
    padding: tuple,
    value: float = 0.0,
) -> np.ndarray:
    input_size = tuple(input_tensor.shape)
    max_pad = Graph(
        "pad",
        lambda x: pad_tensor(x, padding, value=value),
        input_types=[TensorType(DType.float32, input_size)],
    )

    session = InferenceSession()
    model = session.load(max_pad)
    result = model.execute(input_tensor)[0]
    return result.to_numpy()


def test_pad_3d(RTOL, init_np_tensor):
    # Test parameters
    batch_size = 2
    seq_length = 3
    channels = 4
    padding = (1, 1, 2, 2, 0, 0)  # pad last two dims
    value = 0.0

    # Initialize input tensor
    input_size = (batch_size, seq_length, channels)
    input_tensor = init_np_tensor(size=input_size)

    # PyTorch reference implementation
    pt_input = torch.from_numpy(input_tensor)
    pt_output = F.pad(pt_input, padding, mode="constant", value=value).numpy()

    # MAX implementation
    max_output = get_max_pad_result(
        input_tensor=input_tensor,
        padding=padding,
        value=value,
    )

    np.testing.assert_allclose(pt_output, max_output, rtol=RTOL)


def test_pad_no_padding(RTOL, init_np_tensor):
    # Test parameters
    batch_size = 2
    seq_length = 3
    channels = 4
    padding = (0, 0, 0, 0, 0, 0)
    value = 0.0

    # Initialize input tensor
    input_size = (batch_size, seq_length, channels)
    input_tensor = init_np_tensor(size=input_size)

    # PyTorch reference implementation
    pt_input = torch.from_numpy(input_tensor)
    pt_output = F.pad(pt_input, padding, mode="constant", value=value).numpy()

    # MAX implementation
    max_output = get_max_pad_result(
        input_tensor=input_tensor,
        padding=padding,
        value=value,
    )

    np.testing.assert_allclose(pt_output, max_output, rtol=RTOL)
