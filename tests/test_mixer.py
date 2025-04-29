import numpy as np
import pytest
import torch
from max.dtype import DType
from max.engine.api import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
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


@pytest.fixture
def RTOL():
    return 1e-6


@pytest.fixture
def init_np_tensor():
    def _init_np_tensor(size, dtype=np.float32):
        return np.random.rand(*size).astype(dtype)

    return _init_np_tensor


@pytest.fixture
def max_device(device=None):
    return device if device else DeviceRef.CPU()


def get_max_mixer_results(
    device,
    config: Mamba2Config,
    batch_size: int,
    seq_len: int,
    hidden_states: np.ndarray,
    attention_mask: np.ndarray,
    layer_idx: int = 1,
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
    model = session.load(
        mixer_graph,
        weights_registry=mamba2_mixer_initializer(config=config)
        | mamba2_mixer_random_initializer(config=config)
        | mamba2_cache_initializer(batch_size=batch_size, config=config),
    )
    return model.execute(hidden_states, attention_mask)


def get_hf_mixer_results(
    config: HF_MAMBA2CFG,
    batch_size: int,
    seq_len: int,
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
            cache_position=list(range(0, batch_size)),
        )
        .detach()
        .numpy()
    )
    return ctx_states, ctx_states_masked, ctx_states_cache


@pytest.mark.parametrize("batch_size,seq_len", [(2, 4), (1, 2)])
def test_mamba2_mixer_equivalence(
    RTOL, max_device, init_np_tensor, batch_size, seq_len
):
    # MAX config
    max_config = Mamba2Config()
    hidden_size = (batch_size, seq_len, max_config.hidden_size)
    attention_mask_shape = (batch_size, seq_len)
    hidden_states = 1 / np.sqrt(max_config.hidden_size) * init_np_tensor(hidden_size)
    hidden_states = hidden_states.astype(np.float32)
    attention_mask = np.ceil(np.round(init_np_tensor(attention_mask_shape)))
    attention_mask = attention_mask.astype(np.float32)
    ctx_states, ctx_states_masked, ctx_states_cache = get_max_mixer_results(
        device=max_device,
        config=max_config,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_states=hidden_states.astype(np.float32),
        attention_mask=attention_mask.astype(np.float32),
    )
    # HF config
    hf_config = HF_MAMBA2CFG()
    ctx_states_hf, ctx_states_masked_hf, ctx_states_cache_hf = get_hf_mixer_results(
        config=hf_config,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
    )
    # Only compare shapes and types for now, as weights are random and not aligned
    assert ctx_states.to_numpy().shape == ctx_states_hf.shape
    assert ctx_states_masked.to_numpy().shape == ctx_states_masked_hf.shape
    assert ctx_states_cache.to_numpy().shape == ctx_states_cache_hf.shape
    # Optionally: check dtype
    assert ctx_states.to_numpy().dtype == ctx_states_hf.dtype
    assert ctx_states_masked.to_numpy().dtype == ctx_states_masked_hf.dtype
    assert ctx_states_cache.to_numpy().dtype == ctx_states_cache_hf.dtype
