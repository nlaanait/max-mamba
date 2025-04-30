from dataclasses import dataclass
from typing import Optional

from max import nn
from max.graph import DeviceRef, TensorValue
from max.nn.layer import LayerList

from max_mamba.config import Mamba2Config
from max_mamba.layers import Mamba2Block, Mamba2Cache, MambaRMSNormGated


@dataclass
# mirrors transformers.models.mamba2.modeling_mamba2
class Mamba2Output:
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
        input_ids: Optional[TensorValue] = None,
        inputs_embeds: Optional[TensorValue] = None,
        cache_params: Optional[Mamba2Cache] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[list[int]] = None,
        attention_mask: Optional[TensorValue] = None,
    ) -> Mamba2Output:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else False

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

        return Mamba2Output(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
            hidden_states=all_hidden_states,  # type: ignore
        )
