from typing import Any
from max import engine
from max.dtype import DType
from max.graph import Graph, TensorType, ops
from max.graph.weights import load_weights
from max.driver import Tensor
from max_mamba.layers import RMSNormGated
import numpy as np

from transformers.models.mamba2.modeling_mamba2 import MambaRMSNormGated
import torch

torch.manual_seed(1234)


def init_hidden_state(hidden_size: tuple[int] = (8, 16)) -> torch.Tensor:
    return torch.rand(hidden_size)


def rmsnorm_gated_test():
    hidden_size = (8, 16)
    hidden_state = init_hidden_state(hidden_size=hidden_size)
    pt_rms_gated = MambaRMSNormGated(hidden_size=hidden_size)
    pt_output = pt_rms_gated.forward(hidden_state).detach().numpy()
    print(f"torch output: {pt_output}")
    max_output = max_rms_gated(hidden_state=hidden_state)
    print(f"max output: {max_output}")
    assert np.allclose(pt_output, max_output)


def max_rms_gated(hidden_state: np.ndarray) -> np.ndarray:
    # 1. Build the graph
    hidden_size = tuple(hidden_state.shape)
    # weights = np.ones_like(hidden_state, dtype=hidden_state.dtype)
    # weights = Tensor.from_numpy(weights)
    # input_type = TensorType(dtype=DType.float32, shape=hidden_size)
    print(hidden_size)
    max_rms_gated = Graph(
        "rmsnorm_gated",
        RMSNormGated(hidden_size=hidden_size),
        input_types=[TensorType(DType.float32, hidden_size)],
    )

    session = engine.InferenceSession()
    model = session.load(max_rms_gated)

    for tensor in model.input_metadata:
        print(f"name: {tensor.name}, shape: {tensor.shape}, dtype: {tensor.dtype}")

    # 3. Execute the graph
    ret = model.execute(hidden_state)[0]
    print("result:", ret)
    ret_max = ret.to_numpy()
    return ret_max


if __name__ == "__main__":
    rmsnorm_gated_test()
