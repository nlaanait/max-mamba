import numpy as np
import pytest
from max.dtype import DType
from max.engine.api import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from transformers.models.mamba2.modeling_mamba2 import (
    MambaRMSNormGated as HF_MambaRMSNormGated,
)

from max_mamba.layers import MambaRMSNormGated


@pytest.mark.parametrize("hidden_size", [(8, 16), (16, 32), (32, 64), (64, 128)])
def test_rmsnorm_gated(hidden_size, RTOL, max_device, init_pt_tensor):
    hidden_state = init_pt_tensor(size=hidden_size)

    # PyTorch implementation
    pt_rms_gated = HF_MambaRMSNormGated(hidden_size=hidden_size[-1])
    pt_output = pt_rms_gated.forward(hidden_state).detach().numpy()

    # MAX implementation
    max_output = get_max_rms_gated(hidden_state=hidden_state.numpy(), device=max_device)

    np.testing.assert_allclose(pt_output, max_output, rtol=RTOL)


def get_max_rms_gated(hidden_state: np.ndarray, device: DeviceRef) -> np.ndarray:
    hidden_size = tuple(hidden_state.shape)
    max_rms_gated = Graph(
        "rmsnorm_gated",
        MambaRMSNormGated(hidden_size=hidden_size[-1]),
        input_types=[TensorType(DType.float32, hidden_size, device=device)],
    )

    session = InferenceSession()
    model = session.load(
        max_rms_gated, weights_registry={"weight": np.ones_like(hidden_state)}
    )
    return model.execute(hidden_state)[0].to_numpy()
