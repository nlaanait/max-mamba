from dataclasses import dataclass
from typing import Optional

import torch
from max import nn
from max.graph import DeviceRef, TensorValue
from max.nn.layer import LayerList
from transformers.generation.utils import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
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

    config_class = Mamba2Config
    base_model_prefix = "backbone"
    _no_split_modules = ["Mamba2Block"]
    supports_gradient_checkpointing = (
        False  # Max doesn't support gradient checkpointing yet
    )
    _is_stateful = True

    def __init__(self, PretrainedConfig, *inputs, **kwargs):
        super().__init__(PretrainedConfig)
        self.device = kwargs.get("device", DeviceRef.CPU())

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
    def __init__(self, config: Mamba2Config, device: DeviceRef) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_size,
            device=device,
            dtype=config.dtype,
        )
        self.layers = LayerList(
            [
                Mamba2Block(config=config, layer_idx=idx)
                for idx in range(config.num_hidden_layers)
            ]
        )
        self.norm_f = MambaRMSNormGated(
            hidden_size=config.hidden_size, eps=config.layer_norm_epsilon
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
        device: DeviceRef,
    ):
        super().__init__(config)
        self.backbone = Mamba2Model(config=config, device=device)
        self.lm_head = nn.Linear(
            in_dim=config.hidden_size,
            out_dim=config.vocab_size,
            dtype=config.dtype,
            device=device,
            has_bias=False,
        )
        self.post_init()

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embedings(self, new_embeddings: nn.Module):
        self.lm_head = new_embeddings

    def get_input_embeddings(self) -> nn.Module:
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings: nn.Module):
        return self.backbone.set_input_embeddings(new_embeddings)

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

    def forward(
        self,
        input_ids: TensorValue | None = None,
        inputs_embeds: TensorValue | None = None,
        cache_params: Mamba2Cache | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        use_cache: bool | None = None,
        cache_position: list[int] | None = None,
        attention_mask: TensorValue | None = None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        mamba2_outputs = self.backbone(
            input_ids,
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )
        hidden_states = mamba2_outputs[0]

        logits = self.lm_head(hidden_states)

        if not return_dict:
            output = (logits,) + mamba2_outputs[1:]
            return output

        return Mamba2ForCausalLMOutput(
            logits=logits,
            cache_params=mamba2_outputs.cache_params,
            hidden_states=mamba2_outputs.hidden_states,
        )
