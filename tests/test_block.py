import numpy as np
import pytest
import torch
from max.dtype import DType
from max.engine.api import InferenceSession
from max.graph import Graph, TensorType
from transformers.models.mamba2.modeling_mamba2 import Mamba2Block as HF_Mamba2Block
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache as HF_Mamba2Cache

from max_mamba.layers import Mamba2Block, Mamba2Cache, mamba2_cache_initializer
from max_mamba.layers.mixer import (
    mamba2_mixer_initializer,
    mamba2_mixer_random_initializer,
)


def get_max_block_results(
    device,
    config,
    batch_size: int,
    seq_len: int,
    hidden_states: np.ndarray,
    attention_mask: np.ndarray,
    layer_idx: int = 1,
    weights_registry: dict | None = None,
):
    hidden_size = (batch_size, seq_len, config.hidden_size)
    attention_mask_shape = (batch_size, seq_len)
    with Graph(
        "mamba2_block",
        input_types=(
            TensorType(DType.float32, hidden_size, device),
            TensorType(DType.float32, attention_mask_shape, device),
        ),
    ) as block_graph:
        cache = Mamba2Cache(config=config, batch_size=batch_size)
        hidden_states_in, attention_mask_in = block_graph.inputs
        block = Mamba2Block(config=config, layer_idx=layer_idx)
        block.state_dict()

        # Test all variations of forward calls
        states = block(hidden_states=hidden_states_in)
        states_masked = block(
            hidden_states=hidden_states_in, attention_mask=attention_mask_in
        )
        states_cache = block(
            hidden_states=hidden_states_in,
            attention_mask=attention_mask_in,
            cache_params=cache,
            cache_position=list(range(0, batch_size)),
        )
        block_graph.output(states, states_masked, states_cache)

    session = InferenceSession()
    if weights_registry is None:
        weights_registry = (
            mamba2_mixer_initializer(config=config)
            | mamba2_mixer_random_initializer(config=config)
            | mamba2_cache_initializer(batch_size=batch_size, config=config)
        )
    model = session.load(block_graph, weights_registry=weights_registry)
    return model.execute(hidden_states, attention_mask)


def get_hf_block_results(
    config,
    batch_size: int,
    hidden_states: np.ndarray,
    attention_mask: np.ndarray,
    layer_idx: int = 1,
):
    hidden_states_pt = torch.from_numpy(hidden_states)
    attention_mask_pt = torch.from_numpy(attention_mask)
    block = HF_Mamba2Block(config=config, layer_idx=layer_idx)

    # Get results for all variations
    states = block(hidden_states=hidden_states_pt).detach().numpy()
    states_masked = (
        block(hidden_states=hidden_states_pt, attention_mask=attention_mask_pt)
        .detach()
        .numpy()
    )

    cache = HF_Mamba2Cache(config=config, batch_size=batch_size)
    states_cache = (
        block(
            hidden_states=hidden_states_pt,
            attention_mask=attention_mask_pt,
            cache_params=cache,
            cache_position=list(range(0, batch_size)),
        )
        .detach()
        .numpy()
    )

    hf_weights = {k: v.detach().numpy() for k, v in block.state_dict().items()}
    return states, states_masked, states_cache, hf_weights


@pytest.mark.parametrize("batch_size,seq_len", [(1, 2), (2, 5)])
def test_mamba2_block_equivalence(
    RTOL, max_device, init_np_tensor, batch_size, seq_len, mamba2_configs
):
    # Get configs
    max_config, hf_config = mamba2_configs

    # Initialize test data
    hidden_size = (batch_size, seq_len, max_config.hidden_size)
    attention_mask_shape = (batch_size, seq_len)
    hidden_states = init_np_tensor(hidden_size)
    hidden_states = hidden_states.astype(np.float32)
    attention_mask = np.ceil(np.round(init_np_tensor(attention_mask_shape)))
    attention_mask = attention_mask.astype(np.float32)

    # Get HF implementation results
    states_hf, states_masked_hf, states_cache_hf, hf_weights = get_hf_block_results(
        config=hf_config,
        batch_size=batch_size,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
    )

    # Run MAX with HF weights
    weights_registry = (
        mamba2_mixer_initializer(config=max_config, parent="mixer")
        | hf_weights
        | mamba2_cache_initializer(config=max_config, batch_size=batch_size)
    )
    for k, v in weights_registry.items():
        if k == "mixer.conv1d.weight":
            weights_registry[k] = np.ascontiguousarray(
                np.transpose(v, axes=(2, 1, 0))
            )  # NCHW --> NHWC

    states, states_masked, states_cache = get_max_block_results(
        device=max_device,
        config=max_config,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        weights_registry=weights_registry,
    )

    # Compare results
    np.testing.assert_allclose(states.to_numpy(), states_hf, rtol=RTOL, atol=RTOL)
    np.testing.assert_allclose(
        states_masked.to_numpy(), states_masked_hf, rtol=RTOL, atol=RTOL
    )
    np.testing.assert_allclose(
        states_cache.to_numpy(), states_cache_hf, rtol=RTOL, atol=RTOL
    )
