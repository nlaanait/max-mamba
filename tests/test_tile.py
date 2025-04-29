from typing import Optional

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from conftest import RTOL, init_np_tensor
from max.dtype import DType
from max.engine.api import InferenceSession
from max.graph import DeviceRef, Graph, TensorType

from max_mamba.ops import tile_tensor


def get_max_tile_result(
    input_tensor: np.ndarray, dims: tuple, device: DeviceRef
) -> np.ndarray:
    input_size = tuple(input_tensor.shape)
    max_pad = Graph(
        "tile",
        lambda x: tile_tensor(x, dims=dims),
        input_types=[TensorType(DType.float32, input_size, device=device)],
    )

    session = InferenceSession()
    model = session.load(max_pad)
    result = model.execute(input_tensor)[0]
    return result.to_numpy()


@pytest.mark.parametrize(
    "dims",
    [
        (1,),
        (4,),
        (1, 2),
        (1, 1, 2),
        (2, 1, 2, 4),
    ],
)
def test_tile(RTOL, max_device, init_np_tensor, dims):
    # Test parameters
    batch_size = 2
    seq_length = 3
    channels = 4

    # Initialize input tensor
    input_size = (batch_size, seq_length, channels)
    input_tensor = init_np_tensor(size=input_size)

    # PyTorch reference implementation
    pt_input = torch.from_numpy(input_tensor)
    pt_output = torch.tile(
        pt_input,
        dims=dims,
    )

    # MAX implementation
    max_output = get_max_tile_result(
        input_tensor=input_tensor, dims=dims, device=max_device
    )
    np.testing.assert_allclose(pt_output, max_output, rtol=RTOL)
