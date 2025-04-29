import numpy as np
import pytest
import torch
from conftest import RTOL, init_np_tensor
from max.dtype import DType
from max.engine.api import InferenceSession
from max.graph import DeviceRef, Graph, TensorType

from max_mamba.ops import roll_tensor


def get_max_roll_result(
    input_tensor: np.ndarray, shifts, dims, device: DeviceRef
) -> np.ndarray:
    input_size = tuple(input_tensor.shape)
    max_graph = Graph(
        "roll",
        lambda x: roll_tensor(x, shifts=shifts, dims=dims),
        input_types=[TensorType(DType.float32, input_size, device=device)],
    )
    session = InferenceSession()
    model = session.load(max_graph)
    result = model.execute(input_tensor)[0]
    return result.to_numpy()


@pytest.mark.parametrize(
    "shifts,dims",
    [
        (0, 0),
        (1, 0),
        (-1, 0),
        (1, 1),
        (2, 0),
        ((1, 1), (0, 1)),
        ((-1, 2), (0, 2)),
    ],
)
def test_roll(RTOL, max_device, init_np_tensor, shifts, dims):
    # Test parameters
    batch_size = 2
    seq_length = 3
    channels = 4

    input_size = (batch_size, seq_length, channels)
    input_tensor = init_np_tensor(size=input_size)

    # PyTorch reference
    pt_input = torch.from_numpy(input_tensor)
    pt_output = torch.roll(pt_input, shifts=shifts, dims=dims)

    # MAX implementation
    max_output = get_max_roll_result(input_tensor, shifts, dims, device=max_device)
    np.testing.assert_allclose(pt_output, max_output, rtol=RTOL)
