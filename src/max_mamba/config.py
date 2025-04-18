from dataclasses import dataclass
from typing import Callable, Literal, Optional

from max.dtype import DType
from max.graph import DeviceRef, TensorValue
from max.graph.quantization import QuantizationConfig, QuantizationEncoding
from max.pipelines.config import MAXModelConfig, PipelineConfig

# from max.pipelines.max_config import MAXModelConfig, MAXModelConfigBase
from max.pipelines.max_config import MAXConfig
from transformers import Mamba2Config as HF_MAMBA2CFG


@dataclass
class MambaConfigBase(MAXConfig):
    """Base configuration for Mamba models."""

    num_heads: int = 128
    head_dim: int = 64
    vocab_size: int = 32768
    hidden_size: int = 4096
    state_size: int = 128
    num_hidden_layers: int = 64
    layer_norm_epsilon: float = 1e-5
    max_seq_len: int = 2**16
    expand: int = 2
    conv_kernel: int = 4
    n_groups: int = 8
    use_bias: bool = False
    use_conv_bias: bool = True
    hidden_act: str = "silu"
    initializer_range: float = 0.1
    residual_in_fp32: bool = True
    time_step_rank: str = "auto"
    time_step_min: float = 0.001
    time_step_max: float = 0.1
    time_step_floor: float = 1e-4
    time_step_limit: tuple[float, ...] = (0.0, float("inf"))
    rescale_prenorm_residual: bool = False
    use_cache: bool = True
    rms_norm: bool = True
    chunk_size: int = 256
    tie_word_embeddings: bool = False
    # dtype: DType = DType.bfloat16
    dtype: DType = DType.float32
    model_quantization_encoding: Optional[QuantizationEncoding] = None
    quantization_config: Optional[QuantizationConfig] = None
    logits_postprocessor: Callable[[TensorValue], TensorValue] | None = None
    all_logits: bool = True
    norm_method: Literal["rms_norm_gated"] = "rms_norm_gated"
    devices: list[DeviceRef] | None = None

    def help(self) -> dict[str, str]:
        return {}


@dataclass
class Mamba2Config(MambaConfigBase):

    # TODO: implement logic
    @staticmethod
    def calculate_max_seq_len(
        pipeline_config: PipelineConfig | None, huggingface_config: HF_MAMBA2CFG
    ) -> int:
        return 2**16

    @staticmethod
    def generate(
        pipeline_config: PipelineConfig | None,  # TODO: remove None
        huggingface_config: HF_MAMBA2CFG,
        dtype: DType,
        logits_postprocessor: Callable[[TensorValue], TensorValue] | None,
        norm_method: Literal["rms_norm_gated"] = "rms_norm_gated",
    ):
        if pipeline_config:
            device_refs = [
                DeviceRef(spec.device_type, spec.id)
                for spec in pipeline_config.model_config.device_specs
            ]
            return Mamba2Config(
                hidden_size=huggingface_config.hidden_size,
                num_hidden_layers=huggingface_config.num_hidden_layers,
                vocab_size=huggingface_config.vocab_size,
                dtype=dtype,
                model_quantization_encoding=pipeline_config.model_config.graph_quantization_encoding,
                quantization_config=pipeline_config.model_config._quant_config,
                all_logits=pipeline_config.enable_echo,
                max_seq_len=Mamba2Config().calculate_max_seq_len(
                    pipeline_config, huggingface_config=huggingface_config
                ),
                norm_method=norm_method,
                logits_postprocessor=logits_postprocessor,
                devices=device_refs,
            )

        return Mamba2Config(
            hidden_size=huggingface_config.hidden_size,
            num_hidden_layers=huggingface_config.num_hidden_layers,
            vocab_size=huggingface_config.vocab_size,
            dtype=dtype,
            max_seq_len=Mamba2Config.calculate_max_seq_len(
                pipeline_config, huggingface_config=huggingface_config
            ),
            norm_method=norm_method,
            logits_postprocessor=logits_postprocessor,
        )
