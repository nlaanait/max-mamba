import numpy as np
import pytest
import torch
from max.graph import DeviceRef
from transformers import Mamba2Config as HF_MAMBA2CFG

from max_mamba.config import Mamba2Config


@pytest.fixture(scope="session")
def torch_seed():
    torch.manual_seed(1234)
    np.random.seed(1234)


@pytest.fixture
def init_pt_tensor():
    def _init_pt_tensor(size=(8, 16), dtype=torch.float32):
        return torch.rand(size=size, dtype=dtype)

    return _init_pt_tensor


@pytest.fixture
def init_np_tensor():
    def _init_np_tensor(size, dtype=np.float32):
        return np.random.rand(*size).astype(dtype)

    return _init_np_tensor


@pytest.fixture
def mamba2_configs():
    max_config = Mamba2Config()
    hf_config = HF_MAMBA2CFG()
    return max_config, hf_config


@pytest.fixture
def RTOL():
    return 1e-5


@pytest.fixture
def max_device(device=None):
    return device if device else DeviceRef.CPU()
