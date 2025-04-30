import numpy as np
import pytest
import torch
from max.dtype import DType
from max.engine.api import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from transformers import Mamba2Config as HF_MAMBA2CFG
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache as HF_Mamba2Cache

from max_mamba.config import Mamba2Config
from max_mamba.layers import Mamba2Cache


def get_hf_cache_results(
    batch_size: int,
    config: HF_MAMBA2CFG,
    new_conv_state: torch.Tensor,
    new_ssm_state: torch.Tensor,
    layer_idx: int,
    dtype: torch.dtype = torch.float32,
) -> tuple[np.ndarray, ...]:
    pt_cache = HF_Mamba2Cache(config, batch_size=batch_size, dtype=dtype)
    pt_cache_state_pre_init = pt_cache.update_conv_state(
        layer_idx=layer_idx, new_conv_state=new_conv_state, cache_init=True
    )
    pt_cache_state_post_init = pt_cache.update_conv_state(
        layer_idx=layer_idx - 1,
        new_conv_state=torch.transpose(pt_cache_state_pre_init, 1, 2).contiguous(),
        cache_init=False,
    )
    pt_cache_ssm_state = pt_cache.update_ssm_state(
        layer_idx=layer_idx, new_ssm_state=new_ssm_state
    )

    return (
        pt_cache_state_pre_init.numpy(),
        pt_cache_state_post_init.numpy(),
        pt_cache_ssm_state.numpy(),
    )


def get_max_cache_results(
    batch_size: int,
    config: Mamba2Config,
    new_conv_state: np.ndarray,
    new_ssm_state: np.ndarray,
    layer_idx: int,
    device: DeviceRef,
):
    input_size = new_conv_state.shape
    conv_cache_size = (config.num_hidden_layers,) + input_size
    ssm_cache_size = (
        config.num_hidden_layers,
        batch_size,
        config.num_heads,
        config.head_dim,
        config.state_size,
    )
    init_conv_states = np.zeros(conv_cache_size, dtype=np.float32)
    init_ssm_states = np.zeros(ssm_cache_size, dtype=np.float32)
    with Graph(
        "mamba2_cache",
        input_types=(
            TensorType(DType.float32, new_conv_state.shape, device=device),
            TensorType(DType.float32, new_ssm_state.shape, device=device),
        ),
    ) as cache_graph:
        conv_state, ssm_state = cache_graph.inputs
        cache = Mamba2Cache(config=config, batch_size=batch_size)
        conv_state_pre_init = cache.update_conv_state(
            layer_idx=layer_idx, new_conv_state=conv_state, cache_init=True  # type: ignore
        )
        conv_state_post_init = cache.update_conv_state(
            layer_idx=layer_idx - 1, new_conv_state=conv_state_pre_init.transpose(1, 2), cache_init=False  # type: ignore
        )
        ssm_state = cache.update_ssm_state(layer_idx=layer_idx, new_ssm_state=ssm_state)  # type: ignore
        cache_graph.output(conv_state_pre_init, conv_state_post_init, ssm_state)

    sess = InferenceSession()
    cache = sess.load(
        cache_graph,
        weights_registry={
            "conv_states": init_conv_states,
            "ssm_states": init_ssm_states,
        },
    )

    conv_state_pre_init, conv_state_post_init, ssm_state = cache.execute(
        new_conv_state.astype(np.float32), new_ssm_state.astype(np.float32)
    )
    return (
        conv_state_pre_init.to_numpy(),
        conv_state_post_init.to_numpy(),
        ssm_state.to_numpy(),
    )


def test_cache(RTOL, max_device, init_np_tensor, mamba2_configs):
    max_config, hf_config = mamba2_configs
    batch_size = 1
    # Initialize test data
    conv_state_size = (
        batch_size,
        int(hf_config.expand * hf_config.hidden_size)
        + 2 * hf_config.n_groups * hf_config.state_size,
        hf_config.conv_kernel,
    )
    ssm_state_size = (
        batch_size,
        hf_config.num_heads,
        hf_config.head_dim,
        hf_config.state_size,
    )
    layer_idx = 1

    conv_state = init_np_tensor(size=conv_state_size)
    ssm_state = init_np_tensor(size=ssm_state_size)
    pt_conv_state = torch.from_numpy(conv_state)
    pt_ssm_state = torch.from_numpy(ssm_state)

    # Get results from both implementations
    conv_init_pt, conv_post_pt, ssm_pt = get_hf_cache_results(
        batch_size=batch_size,
        config=hf_config,
        new_conv_state=pt_conv_state,
        new_ssm_state=pt_ssm_state,
        layer_idx=layer_idx,
    )

    conv_init_max, conv_post_max, ssm_max = get_max_cache_results(
        batch_size=batch_size,
        config=max_config,
        new_conv_state=conv_state,
        new_ssm_state=ssm_state,
        layer_idx=layer_idx,
        device=max_device,
    )

    # Compare results
    np.testing.assert_allclose(conv_init_pt, conv_init_max, rtol=RTOL)
    np.testing.assert_allclose(conv_post_pt, conv_post_max, rtol=RTOL)
    np.testing.assert_allclose(ssm_pt, ssm_max, rtol=RTOL)
