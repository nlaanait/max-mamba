from typing import Optional

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from conftest import RTOL, init_np_tensor
from max.dtype import DType
from max.engine.api import InferenceSession
from max.graph import Graph, TensorType

from max_mamba.ops import clamp_tensor


def get_max_clamp_result(
    input_tensor: np.ndarray,
    min_value: Optional[float],
    max_value: Optional[float],
) -> np.ndarray:
    input_size = tuple(input_tensor.shape)
    max_pad = Graph(
        "clamp",
        lambda x: clamp_tensor(x, min_val=min_value, max_val=max_value),
        input_types=[TensorType(DType.float32, input_size)],
    )

    session = InferenceSession()
    model = session.load(max_pad)
    result = model.execute(input_tensor)[0]
    return result.to_numpy()


@pytest.mark.parametrize(
    "min_value, max_value",
    [
        (0.5, None),
        (None, 0.5),
        (0.5, 0.5),
    ],
)
def test_clamp(RTOL, init_np_tensor, min_value, max_value):
    # Test parameters
    batch_size = 2
    seq_length = 3
    channels = 4

    # Initialize input tensor
    input_size = (batch_size, seq_length, channels)
    input_tensor = init_np_tensor(size=input_size)

    # PyTorch reference implementation
    pt_input = torch.from_numpy(input_tensor)
    pt_output = torch.clamp(
        pt_input,
        max=max_value,
        min=min_value,
    )

    # MAX implementation
    max_output = get_max_clamp_result(
        input_tensor=input_tensor,
        max_value=max_value,
        min_value=min_value,
    )

    np.testing.assert_allclose(pt_output, max_output, rtol=RTOL)
