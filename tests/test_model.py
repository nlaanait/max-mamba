import numpy as np
import pytest
import torch
from max.dtype import DType
from max.engine.api import InferenceSession
from max.graph import Graph, TensorType
from transformers import AutoTokenizer
from transformers.models.mamba2 import Mamba2Model as HF_Mamba2Model

from max_mamba import Mamba2Model
from max_mamba.layers.cache import mamba2_cache_initializer
from max_mamba.layers.mixer import mamba2_mixer_initializer


def get_max_model_results(
    device,
    config,
    input_ids: np.ndarray,
    weights_registry: dict | None = None,
):
    input_ids_size = tuple(input_ids.shape)
    with Graph(
        "Mamba2Model",
        input_types=(TensorType(DType.int64, input_ids_size, device),),
    ) as model_graph:
        input_ids_t = model_graph.inputs[0]
        model = Mamba2Model(config=config, device=device)
        model.state_dict()
        model_graph.output(
            model(input_ids=input_ids_t, return_dict=False).last_hidden_state
        )

    session = InferenceSession()
    if weights_registry is None:
        weights_registry = {}
        for i in range(config.num_hidden_layers):
            weights_registry |= mamba2_mixer_initializer(
                config=config, parent=f"layers.{i}.mixer"
            )
        weights_registry |= mamba2_cache_initializer(config=config, batch_size=1)

    model = session.load(model_graph, weights_registry=weights_registry)
    return model.execute(input_ids)[0]


def get_hf_model_results(config, input_ids: np.ndarray):
    input_ids_pt = torch.from_numpy(input_ids)
    model = HF_Mamba2Model(config=config)
    outputs = model(input_ids=input_ids_pt)
    hf_weights = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    return outputs.last_hidden_state.detach().numpy(), hf_weights


@pytest.mark.parametrize("num_hidden_layers", [1, 2, 4])
def test_mamba2_model_equivalence(RTOL, max_device, num_hidden_layers, mamba2_configs):
    # Get configs and set num layers
    max_config, hf_config = mamba2_configs
    max_config.num_hidden_layers = num_hidden_layers
    hf_config.num_hidden_layers = num_hidden_layers

    # Initialize test data - simple sequence
    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf")
    input_ids = tokenizer("mic test: 1, 2, 3!", return_tensors="np")["input_ids"]
    # input_ids = np.array([[1, 2, 3]], dtype=np.int64)

    # Get HF implementation results
    states_hf, hf_weights = get_hf_model_results(
        config=hf_config,
        input_ids=input_ids,
    )

    # Run MAX with HF weights
    weights_registry = {}
    for i in range(max_config.num_hidden_layers):
        weights_registry |= mamba2_mixer_initializer(
            config=max_config, parent=f"layers.{i}.mixer"
        )
    weights_registry |= hf_weights
    weights_registry |= mamba2_cache_initializer(config=max_config, batch_size=1)

    # Handle conv1d weight transposition
    for k, v in weights_registry.items():
        if "conv1d.weight" in k:
            weights_registry[k] = np.ascontiguousarray(
                np.transpose(v, axes=(2, 1, 0))
            )  # NCHW --> NHWC

    states = get_max_model_results(
        device=max_device,
        config=max_config,
        input_ids=input_ids,
        weights_registry=weights_registry,
    )

    # Compare results
    np.testing.assert_allclose(states.to_numpy(), states_hf, rtol=RTOL, atol=RTOL * 10)
