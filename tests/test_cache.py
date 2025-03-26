from max.engine.api import InferenceSession
from max.dtype import DType
from max.graph import Graph, TensorType
from max_mamba.layers import Mamba2Cache
from max_mamba.config import Mamba2Config
import numpy as np


from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache as HF_Mamba2Cache
from transformers import Mamba2Config as HF_MAMBA2CFG

import torch

torch.manual_seed(1234)


def init_pt_tensor(
    size: tuple[int, ...] = (8, 16), dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    return torch.rand(size=size, dtype=dtype)


def init_np_tensor(size: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    return np.random.rand(*size).astype(dtype)


def init_mamba2configs() -> tuple[Mamba2Config, HF_MAMBA2CFG]:
    max_config = Mamba2Config()
    hf_config = HF_MAMBA2CFG()
    return (max_config, hf_config)


def hf_cache(
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
    # TODO: re-enable after resolving the assigment issue in Transformers
    # pt_cache_state_post_init = pt_cache.update_conv_state(
    #     layer_idx=layer_idx, new_conv_state=pt_cache_state_pre_init, cache_init=False
    # )
    pt_cache_ssm_state = pt_cache.update_ssm_state(
        layer_idx=layer_idx, new_ssm_state=new_ssm_state
    )

    return (
        pt_cache_state_pre_init.numpy(),
        pt_cache_state_pre_init.numpy(),
        pt_cache_ssm_state.numpy(),
    )


def test_cache():
    hf_config = HF_MAMBA2CFG()
    max_config = Mamba2Config.generate(
        pipeline_config=None,
        huggingface_config=hf_config,
        dtype=DType.float32,
        logits_postprocessor=None,
    )
    batch_size = 1
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
    conv_state = init_np_tensor(size=conv_state_size, dtype=np.float32)
    ssm_state = init_np_tensor(size=ssm_state_size, dtype=np.float32)
    pt_conv_state = torch.from_numpy(conv_state)
    pt_ssm_state = torch.from_numpy(ssm_state)
    pt_cache_state_pre_init, pt_cache_state_post_init, pt_cache_ssm_state = hf_cache(
        batch_size=batch_size,
        config=hf_config,
        new_conv_state=pt_conv_state,
        new_ssm_state=pt_ssm_state,
        layer_idx=layer_idx,
    )
    max_cache_state_pre_init, max_cache_state_post_init, max_cache_ssm_state = (
        max_cache(
            batch_size=batch_size,
            config=max_config,
            new_conv_state=conv_state,
            new_ssm_state=ssm_state,
            layer_idx=layer_idx,
        )
    )
    assert np.allclose(pt_cache_state_pre_init, max_cache_state_pre_init)
    assert np.allclose(pt_cache_state_post_init, max_cache_state_post_init)
    assert np.allclose(pt_cache_ssm_state, max_cache_ssm_state)


def max_cache(
    batch_size: int,
    config: Mamba2Config,
    new_conv_state: np.ndarray,
    new_ssm_state: np.ndarray,
    layer_idx: int,
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
            TensorType(DType.float32, new_conv_state.shape),
            TensorType(DType.float32, new_ssm_state.shape),
        ),
    ) as cache_graph:
        conv_state, ssm_state = cache_graph.inputs
        cache = Mamba2Cache(config=config, batch_size=batch_size)
        conv_state_pre_init = cache.update_conv_state(
            layer_idx=layer_idx, new_conv_state=conv_state, cache_init=True  # type: ignore
        )
        conv_state_post_init = cache.update_conv_state(
            layer_idx=layer_idx, new_conv_state=conv_state, cache_init=False  # type: ignore
        )
        ssm_state = cache.update_ssm_state(layer_idx=layer_idx, new_ssm_state=ssm_state)  # type: ignore
        cache.reset()
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
        conv_state_pre_init.to_numpy(),
        ssm_state.to_numpy(),
    )


if __name__ == "__main__":
    test_cache()
