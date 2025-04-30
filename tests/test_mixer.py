from typing import Optional

import numpy as np
import pytest
import torch
from max.dtype import DType
from max.engine.api import InferenceSession
from max.graph import Graph, TensorType
from transformers.models.mamba2 import Mamba2Config as HF_MAMBA2CFG
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache as HF_Mamba2Cache
from transformers.models.mamba2.modeling_mamba2 import Mamba2Mixer as HF_Mamba2Mixer

from max_mamba import Mamba2Config
from max_mamba.layers import (
    Mamba2Cache,
    Mamba2Mixer,
    mamba2_cache_initializer,
    mamba2_mixer_initializer,
    mamba2_mixer_random_initializer,
)


def get_max_mixer_results(
    device,
    config: Mamba2Config,
    batch_size: int,
    seq_len: int,
    hidden_states: np.ndarray,
    attention_mask: np.ndarray,
    layer_idx: int = 1,
    weights_registry: Optional[dict] = None,
):
    hidden_size = (batch_size, seq_len, config.hidden_size)
    attention_mask_shape = (batch_size, seq_len)
    with Graph(
        "mamba2_mixer",
        input_types=(
            TensorType(DType.float32, hidden_size, device),
            TensorType(DType.float32, attention_mask_shape, device),
        ),
    ) as mixer_graph:
        cache = Mamba2Cache(config=config, batch_size=batch_size)
        hidden_states_in, attention_mask_in = mixer_graph.inputs
        mixer = Mamba2Mixer(config=config, layer_idx=layer_idx)
        mixer.state_dict()
        ctx_states = mixer(hidden_states=hidden_states_in)  # type: ignore
        ctx_states_masked = mixer(
            hidden_states=hidden_states_in, attention_mask=attention_mask_in  # type: ignore
        )
        ctx_states_cache = mixer(
            hidden_states=hidden_states_in,  # type: ignore
            attention_mask=attention_mask_in,  # type: ignore
            cache_params=cache,
            cache_position=list(range(0, batch_size)),
        )
        mixer_graph.output(ctx_states, ctx_states_masked, ctx_states_cache)

    session = InferenceSession()
    if weights_registry is None:
        weights_registry = (
            mamba2_mixer_initializer(config=config)
            | mamba2_mixer_random_initializer(config=config)
            | mamba2_cache_initializer(batch_size=batch_size, config=config)
        )
    model = session.load(
        mixer_graph,
        weights_registry=weights_registry,
    )
    return model.execute(hidden_states, attention_mask)


def get_hf_mixer_results(
    config: HF_MAMBA2CFG,
    batch_size: int,
    hidden_states: np.ndarray,
    attention_mask: np.ndarray,
    layer_idx: int = 1,
):
    hidden_states_pt = torch.from_numpy(hidden_states)
    attention_mask_pt = torch.from_numpy(attention_mask)
    mixer = HF_Mamba2Mixer(config=config, layer_idx=layer_idx)
    # w/o attention mask + cache
    ctx_states = mixer.forward(hidden_states=hidden_states_pt).detach().numpy()
    # w/ attention mask
    ctx_states_masked = (
        mixer.forward(hidden_states=hidden_states_pt, attention_mask=attention_mask_pt)
        .detach()
        .numpy()
    )
    # w/ cache
    cache = HF_Mamba2Cache(config=config, batch_size=batch_size)
    ctx_states_cache = (
        mixer.forward(
            hidden_states=hidden_states_pt,
            attention_mask=attention_mask_pt,
            cache_params=cache,
            cache_position=list(range(0, batch_size)),  # type: ignore
        )
        .detach()
        .numpy()
    )
    hf_weights = {k: v.detach().numpy() for k, v in mixer.state_dict().items()}
    return ctx_states, ctx_states_masked, ctx_states_cache, hf_weights


@pytest.mark.parametrize("batch_size,seq_len", [(1, 2), (2, 5)])
def test_mamba2_mixer_equivalence(
    RTOL, max_device, init_np_tensor, batch_size, seq_len, mamba2_configs
):
    # MAX config
    max_config, hf_config = mamba2_configs
    hidden_size = (batch_size, seq_len, max_config.hidden_size)
    attention_mask_shape = (batch_size, seq_len)
    hidden_states = init_np_tensor(hidden_size)
    hidden_states = hidden_states.astype(np.float32)
    attention_mask = np.ceil(np.round(init_np_tensor(attention_mask_shape)))
    attention_mask = attention_mask.astype(np.float32)

    # HF outputs
    ctx_states_hf, ctx_states_masked_hf, ctx_states_cache_hf, hf_weights = (
        get_hf_mixer_results(
            config=hf_config,
            batch_size=batch_size,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
    )

    # # Run MAX with HF weights
    weights_registry = (
        mamba2_mixer_initializer(config=max_config)
        | hf_weights
        | mamba2_cache_initializer(config=max_config, batch_size=batch_size)
    )
    for k, v in weights_registry.items():
        if k == "conv1d.weight":
            weights_registry[k] = np.ascontiguousarray(
                np.transpose(v, axes=(2, 1, 0))
            )  # NCHW --> NHWC

    ctx_states, ctx_states_masked, ctx_states_cache = get_max_mixer_results(
        device=max_device,
        config=max_config,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        weights_registry=weights_registry,
    )

    # # # Compare shapes and dtypes
    assert ctx_states.to_numpy().shape == ctx_states_hf.shape  # type: ignore
    np.testing.assert_allclose(
        ctx_states.to_numpy(), ctx_states_hf, rtol=RTOL, atol=RTOL  # type: ignore
    )
    assert ctx_states_masked.to_numpy().shape == ctx_states_masked_hf.shape  # type: ignore
    np.testing.assert_allclose(
        ctx_states_masked.to_numpy(), ctx_states_masked_hf, rtol=RTOL, atol=RTOL  # type: ignore
    )
    assert ctx_states_cache.to_numpy().shape == ctx_states_cache_hf.shape  # type: ignore
    np.testing.assert_allclose(
        ctx_states_cache.to_numpy(), ctx_states_cache_hf, rtol=RTOL, atol=RTOL  # type: ignore
    )
