import numpy as np
from max.dtype import DType
from max.engine.api import InferenceSession
from max.graph import Graph, TensorType
from transformers.models.mamba2.modeling_mamba2 import MambaRMSNormGated

from max_mamba.layers import RMSNormGated


def test_rmsnorm_gated(RTOL, init_pt_tensor):
    hidden_size = (8, 16)
    hidden_state = init_pt_tensor(size=hidden_size)

    # PyTorch implementation
    pt_rms_gated = MambaRMSNormGated(hidden_size=hidden_size)
    pt_output = pt_rms_gated.forward(hidden_state).detach().numpy()

    # MAX implementation
    max_output = get_max_rms_gated(hidden_state=hidden_state.numpy())

    np.testing.assert_allclose(pt_output, max_output, rtol=RTOL)


def get_max_rms_gated(hidden_state: np.ndarray) -> np.ndarray:
    hidden_size = tuple(hidden_state.shape)
    max_rms_gated = Graph(
        "rmsnorm_gated",
        RMSNormGated(hidden_size=hidden_size),
        input_types=[TensorType(DType.float32, hidden_size)],
    )

    session = InferenceSession()
    model = session.load(
        max_rms_gated, weights_registry={"weight": np.ones_like(hidden_state)}
    )
    return model.execute(hidden_state)[0].to_numpy()
