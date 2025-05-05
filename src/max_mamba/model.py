from dataclasses import dataclass
from typing import Optional

import torch
from max import nn
from max.dtype import DType
from max.engine.api import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.nn.layer import LayerList
from transformers.generation.utils import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.models.mamba2 import Mamba2Config as HF_Mamba2Config
from transformers.utils.generic import ModelOutput

from max_mamba.config import Mamba2Config
from max_mamba.layers import Mamba2Block, Mamba2Cache, MambaRMSNormGated


@dataclass
# mirrors transformers.models.mamba2.modeling_mamba2
class Mamba2Output(ModelOutput):
    """
    Class for the MAMBA2 model outputs.

    Args:
        last_hidden_state (`TensorValue` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        cache_params (`Mamba2Cache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.

            Includes both the State space model state matrices after the selective scan, and the Convolutional states
        hidden_states (`tuple(TensorValue)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `TensorValue` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    last_hidden_state: Optional[TensorValue] = None
    cache_params: Optional[Mamba2Cache] = None
    hidden_states: Optional[tuple[TensorValue]] = None


# mirrors transformers.models.mamba2.modeling_mamba2
class Mamba2CausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss: set to None.
        logits (`TensorValue` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        cache_params (`Mamba2Cache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.

            Includes both the State space model state matrices after the selective scan, and the Convolutional states
        hidden_states (`tuple(TensorValue)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    loss: None = None
    logits: TensorValue | None = None
    cache_params: Mamba2Cache | None = None
    hidden_states: tuple[TensorValue] | None = None


@dataclass
class Mamba2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models. This class adapts the PyTorch-based PreTrainedModel to work with max.nn.
    """

    config_class = HF_Mamba2Config
    base_model_prefix = "backbone"
    _no_split_modules = ["Mamba2Block"]
    supports_gradient_checkpointing = (
        False  # Max doesn't support gradient checkpointing yet
    )
    _is_stateful = True

    # @device.setter
    # def device(self, val):
    #     self._device = val

    def __init__(self, PretrainedConfig, *inputs, **kwargs):
        super().__init__(PretrainedConfig)
        # self.device = kwargs.get("device", DeviceRef.CPU())
        # self.device

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Module):
            module.state_dict()

    def init_weights(self):
        """
        Initializes and prunes weights if needed. Using max.nn initialization methods.
        """
        # Initialize weights for all modules
        self.apply(self._init_weights)

    def post_init(self):
        self.init_weights()

    def get_input_embeddings(self) -> nn.Module:
        """
        Returns the model's input embeddings.

        Returns:
            `nn.Module`: A MAX module mapping vocabulary to hidden states.
        """
        base_model = getattr(self, self.base_model_prefix, self)
        if base_model is not self:
            return base_model.get_input_embeddings()
        else:
            raise NotImplementedError

    def set_input_embeddings(self, value: nn.Module):
        """
        Set model's input embeddings.

        Args:
            value (`nn.Module`): A module mapping vocabulary to hidden states.
        """
        base_model = getattr(self, self.base_model_prefix, self)
        if base_model is not self:
            base_model.set_input_embeddings(value)
        else:
            raise NotImplementedError

    def resize_token_embeddings(self, *args, **kwargs) -> nn.Embedding:
        raise NotImplementedError

    def prune_heads(self, heads_to_prune):
        raise NotImplementedError

    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings if needed.
        """
        if getattr(self.config, "tie_word_embeddings", True):
            output_embeddings = self.get_output_embeddings()
            input_embeddings = self.get_input_embeddings()
            if output_embeddings is not None and input_embeddings is not None:
                self._tie_or_clone_weights(output_embeddings, input_embeddings)

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        """Clone or tie module weights depending if we are using Max's equivalent of TorchScript or not."""
        output_embeddings.weight = input_embeddings.weight


class Mamba2Model(nn.Module):
    def __init__(self, config: Mamba2Config, device: DeviceRef, dtype: DType) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_size,
            device=device,
            dtype=dtype,
        )
        self.layers = LayerList(
            [
                Mamba2Block(config=config, layer_idx=idx, device=device, dtype=dtype)
                for idx in range(config.num_hidden_layers)
            ]
        )
        self.norm_f = MambaRMSNormGated(
            hidden_size=config.hidden_size,
            eps=config.layer_norm_epsilon,
            device=device,
            dtype=dtype,
        )
        self.config = config
        self.device = device if device else DeviceRef.CPU()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    def __call__(
        self,
        input_ids: TensorValue | None = None,
        inputs_embeds: TensorValue | None = None,
        cache_params: Mamba2Cache | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: list[int] | None = None,
        attention_mask: TensorValue | None = None,
    ) -> tuple | Mamba2Output:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else True

        if (input_ids is None) ^ (inputs_embeds is not None):  # ^ is python for xor
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if use_cache:
            if cache_params is None:
                cache_params = Mamba2Cache(
                    config=self.config,
                    batch_size=int(inputs_embeds.shape[0]),
                    device=inputs_embeds.device,
                )
                cache_position = list(range(0, self.config.conv_kernel))
            elif cache_position is None:
                raise ValueError(
                    "cache_position must be specified with `use_cache=True`"
                )
        else:
            cache_params = None

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        for mixer_block in self.layers:
            hidden_states = mixer_block(
                hidden_states=hidden_states,
                cache_params=cache_params,
                cache_position=cache_position,
                attention_mask=attention_mask,
            )
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)  # type: ignore

        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)  # type: ignore

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, cache_params, all_hidden_states]
                if v is not None
            )
        return Mamba2Output(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
            hidden_states=all_hidden_states,  # type: ignore
        )


class Mamba2ForCausalLM(Mamba2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = []

    def __init__(
        self,
        config: Mamba2Config,
        device: DeviceRef = DeviceRef.CPU(),
        dtype: DType = DType.float32,
        batch_size: int = 1,
        max_seq_len: int = 32,
    ):
        super().__init__(config)
        self._device = device
        self._dtype = dtype
        self.config = config
        self._dtype = dtype
        self.session = InferenceSession()
        self._state_dict = {}
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.model = None
        # self.model = self.load_model()

    def _load_model(self) -> Model:
        """Build graph and load model into inference session.

        Returns:
            Model: The compiled model loaded in the inference session.
        """

        input_types = [
            TensorType(
                DType.int64,
                shape=[self.batch_size, self.max_seq_len],
                device=self._device,
            ),
            TensorType(
                DType.float32,
                shape=[self.batch_size, self.max_seq_len],
                device=self._device,
            ),
        ]

        with Graph("mamba2_causal_lm", input_types=input_types) as graph:
            input_ids, attention_mask = graph.inputs

            backbone = Mamba2Model(
                config=self.config, device=self._device, dtype=self._dtype
            )
            backbone.state_dict()
            lm_head = nn.Linear(
                in_dim=self.config.hidden_size,
                out_dim=self.config.vocab_size,
                dtype=self._dtype,
                device=self._device,
                has_bias=False,
            )
            lm_head.state_dict()

            hidden_states = backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=False,
            )[0]
            logits = lm_head(hidden_states)

            graph.output(hidden_states, logits)

            # Save references and state dict
            self.backbone = backbone
            self.lm_head = lm_head

        return self.session.load(graph, weights_registry=self.state_dict())

    def forward(
        self,
        input_ids: TensorValue | None = None,
        attention_mask: TensorValue | None = None,
        **kwargs,
    ):
        self.model = self.model if self.model else self._load_model()
        """Forward pass using loaded model."""
        hidden_states, logits = self.model.execute(input_ids, attention_mask)

        if not self.config.use_return_dict:
            return (logits.to_numpy(), +hidden_states.to_numpy()[1:])

        return Mamba2CausalLMOutput(hidden_states=hidden_states, logits=logits)

    def prepare_inputs_for_generation(
        self,
        input_ids: TensorValue,
        inputs_embeds: TensorValue | None = None,
        use_cache=None,
        cache_params: Mamba2Cache | None = None,
        cache_position: list[int] | None = None,
        attention_mask: TensorValue | None = None,
        **kwargs,
    ):
        if use_cache:
            if cache_position is None:
                raise ValueError(
                    "`cache_position` should not be None as it should have been initialized in "
                    "`model.generate`, you are responsible for passing in a valid `cache_position` if "
                    "you are calling `prepare_inputs_for_generation` directly with `use_cache=True`"
                )

            if isinstance(cache_position, torch.Tensor):
                cache_position = cache_position.detach().tolist()

            if cache_position[0] > 0:
                input_ids = input_ids[:, -1][..., None]

                if attention_mask is not None:
                    attention_mask = None
            else:
                cache_position = list(range(0, self.config.conv_kernel))

        if inputs_embeds is not None and cache_params is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "cache_params": cache_params,
                "use_cache": use_cache,
                "cache_position": cache_position,
            }
        )
        return model_inputs
