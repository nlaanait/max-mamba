import numpy as np
import pytest
import torch
import torch.nn.functional as F
from max.dtype import DType
from max.engine.api import InferenceSession
from max.graph import DeviceRef, Graph, TensorType

from max_mamba.ops import pad_tensor


def get_max_pad_result(
    input_tensor: np.ndarray,
    padding: tuple,
    device: DeviceRef,
    value: float = 0.0,
) -> np.ndarray:
    input_size = tuple(input_tensor.shape)
    max_pad = Graph(
        "pad",
        lambda x: pad_tensor(x, padding, value=value),
        input_types=[TensorType(DType.float32, input_size, device=device)],
    )

    session = InferenceSession()
    model = session.load(max_pad)
    result = model.execute(input_tensor)[0]
    return result.to_numpy()


@pytest.mark.parametrize(
    "batch_size,seq_length,channels,padding,value",
    [
        (2, 3, 4, (1, 1, 2, 2, 0, 0), 0.0),  # original 3d pad test
        (2, 3, 4, (0, 0, 0, 0, 0, 0), 0.0),  # no padding test
        (1, 8, 16, (2, 2, 1, 1, 0, 0), 0.0),  # larger sequence
        (4, 16, 8, (1, 1, 1, 1, 1, 1), 1.0),  # pad all dims with value=1
        (8, 4, 32, (3, 3, 0, 0, 2, 2), -1.0),  # mixed padding with negative value
    ],
)
def test_pad(
    batch_size, seq_length, channels, padding, value, RTOL, max_device, init_np_tensor
):
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
        device=max_device,
    )

    np.testing.assert_allclose(pt_output, max_output, rtol=RTOL)
