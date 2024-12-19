# modeling_canine.py
# Author: Julie Kallini
# Description: This file contains the implementation of the T5 model with
# downsampling via the strided convolution method of Clark et al. 2022.
# https://aclanthology.org/2022.tacl-1.5.pdf
# The code is adapted from HuggingFace's modeling_t5.py. New code sequences
# are labeled with comments.

import torch
import copy
import math
from torch import nn
from models.modeling_t5 import (
    T5LayerCrossAttention,
    T5LayerSelfAttention,
    T5LayerNorm,
    T5LayerFF,
    T5Stack,
    T5Config,
    T5ForConditionalGeneration,
)
from models import shortening
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
)
from transformers.utils import logging
from typing import Optional, Tuple, Union
from dataclasses import dataclass

logger = logging.get_logger(__name__)

@dataclass
class CanineT5BaseModelOutputWithPastAndCrossAttentions(BaseModelOutputWithPastAndCrossAttentions):
    attention_mask: torch.FloatTensor = None

class CanineT5Config(T5Config):
    def __init__(
        self,
        *args,
        downsampling_layer=2,
        use_softmax1=False,
        downsampling_rate=4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.downsampling_layer=downsampling_layer
        self.use_softmax1 = use_softmax1
        self.downsampling_rate = downsampling_rate


TORCH_INIT_FUNCTIONS = {
    "uniform_": nn.init.uniform_,
    "normal_": nn.init.normal_,
    "trunc_normal_": nn.init.trunc_normal_,
    "constant_": nn.init.constant_,
    "xavier_uniform_": nn.init.xavier_uniform_,
    "xavier_normal_": nn.init.xavier_normal_,
    "kaiming_uniform_": nn.init.kaiming_uniform_,
    "kaiming_normal_": nn.init.kaiming_normal_,
    "uniform": nn.init.uniform,
    "normal": nn.init.normal,
    "xavier_uniform": nn.init.xavier_uniform,
    "xavier_normal": nn.init.xavier_normal,
    "kaiming_uniform": nn.init.kaiming_uniform,
    "kaiming_normal": nn.init.kaiming_normal,
}

class CharactersToMolecules(nn.Module):
    """Convert character sequence to initial molecule sequence (i.e. downsample) using strided convolutions."""

    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=config.d_model,
            out_channels=config.d_model,
            kernel_size=config.downsampling_rate,
            stride=config.downsampling_rate,
        )
        self.activation = torch.nn.GELU()
        self.conv.reset_parameters()
        self._init_weights()

    def forward(self, char_encoding: torch.Tensor) -> torch.Tensor:

        # char_encoding has shape [batch, char_seq, hidden_size]
        # We transpose it to be [batch, hidden_size, char_seq]
        char_encoding = torch.transpose(char_encoding, 1, 2)
        downsampled = self.conv(char_encoding)
        downsampled = torch.transpose(downsampled, 1, 2)
        downsampled = self.activation(downsampled)

        return downsampled
    
    def _init_weights(self):
        TORCH_INIT_FUNCTIONS["kaiming_uniform_"](self.conv.weight, a=math.sqrt(5))
        if self.conv.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                TORCH_INIT_FUNCTIONS["uniform_"](self.conv.bias, -bound, bound)


class CanineT5Block(nn.Module):
    """
    Modified version of T5Block that uses CanineT5LayerSelfAttention and
    CanineT5LayerCrossAttention instead of T5LayerSelfAttention and
    T5LayerCrossAttention.
    """

    def __init__(self, config, has_relative_attention_bias=False,
                 #### NEW CODE ####
                 has_downsampling_module=False,
                 #### NEW CODE ####
                 ):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config))

        self.layer.append(T5LayerFF(config))

        #### NEW CODE ####
        # Add downsampling module and layer norm
        self.has_downsampling_module = has_downsampling_module
        self.downsampling_rate = config.downsampling_rate
        if self.has_downsampling_module:
            self.downsampling_module = CharactersToMolecules(config)
            self.down_ln = T5LayerNorm(config.d_model)
        #### NEW CODE ####

    def __get_new_positions_and_mask(self, batch_size, seq_len, boundaries, device):
        # Create filter from boundaries
        keep_this = boundaries > 0.0

        # Calculate the target position for each token
        target_pos = torch.cumsum(keep_this, dim=1) - 1
        new_len = target_pos[:, -1].max().item() + 1

        # Clamp the target position to avoid out of bounds when deleting everything
        target_pos = target_pos.clamp(min=0)

        # Map the positions to the src side. Do this in int32, because it's faster and we will not have sequences
        # longer than 2^31
        positions = torch.arange(seq_len, device=device, dtype=torch.int32).repeat(batch_size, 1)
        positions *= keep_this.int()

        src_side_pos = torch.zeros(batch_size, new_len, device=device, dtype=torch.int32)
        src_side_pos.scatter_add_(1, target_pos, positions)

        return src_side_pos.long()
    
    def __hard_delete_4_dimensions(self, position_bias, positions):
        new_position_bias = torch.gather(position_bias, 1, positions.unsqueeze(2).unsqueeze(3).expand(-1, -1, position_bias.size(2), position_bias.size(3)))
        return new_position_bias
    
    def _downsample_attention_mask(self, char_attention_mask: torch.Tensor):
        """Downsample 2D character attention mask to 2D molecule attention mask using MaxPool1d layer."""

        # first, make char_attention_mask 3D by adding a channel dim
        batch_size, _, _, char_seq_len = char_attention_mask.shape
        poolable_char_mask = torch.reshape(char_attention_mask, (batch_size, 1, char_seq_len))

        # next, apply MaxPool1d to get pooled_molecule_mask of shape (batch_size, 1, mol_seq_len)
        pooled_molecule_mask = torch.nn.MaxPool1d(kernel_size=self.downsampling_rate, stride=self.downsampling_rate)(
            poolable_char_mask.float()
        )

        # finally, squeeze to get tensor of shape (batch_size, mol_seq_len)
        molecule_attention_mask = pooled_molecule_mask.unsqueeze(dim=-2)

        return molecule_attention_mask

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
        #### NEW CODE ####
        input_ids=None,
        #### NEW CODE ####
    ):
        if past_key_value is not None:
            if not self.is_decoder:
                logger.warning(
                    "`past_key_values` is passed to the encoder. Please make sure this is intended.")
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (key / value) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        ##### NEW CODE #####
        if self.has_downsampling_module:

            # Create a simple pad token mask (i.e. inverting back the
            # extended attention mask)
            pad_token_mask = ~(attention_mask.squeeze(-2).squeeze(-2) < 0)
            
            # Compute new position bias
            if position_bias is not None:

                # Keep every nth position bias, where n is the downsampling rate
                keep_this = torch.zeros(pad_token_mask.shape).to(hidden_states.device)
                skip = self.downsampling_rate
                keep_this[:, :-(skip-1):skip] = 1.0
                # keep_this = keep_this * pad_token_mask

                # Compute new token positions
                new_positions = self.__get_new_positions_and_mask(
                    hidden_states.size(0), hidden_states.size(1), keep_this, hidden_states.device)

                # Compute new position bias via deletion
                new_position_bias = self.__hard_delete_4_dimensions(
                    position_bias.permute(0, 2, 3, 1), new_positions)
                new_position_bias = self.__hard_delete_4_dimensions(
                    new_position_bias.permute(0, 2, 1, 3), new_positions)
                position_bias = new_position_bias.permute(0, 3, 2, 1)

            # # Compute new attention mask
            # ones_count = keep_this.sum(dim=1)
            # range_tensor = torch.arange(ones_count.max().item()).unsqueeze(0).to(hidden_states.device)
            # new_attention_mask = torch.where(range_tensor < ones_count.unsqueeze(1), 0.0, torch.finfo(torch.float).min)
            # attention_mask = new_attention_mask.unsqueeze(-2).unsqueeze(-2)

            # Downsample hidden states
            hidden_states = self.downsampling_module(self.down_ln(hidden_states))

            # Downsample attention mask
            attention_mask = self._downsample_attention_mask(attention_mask)

        ##### NEW CODE #####

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        # Keep self-attention outputs and relative position weights
        attention_outputs = self_attention_outputs[2:]

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16:
                clamp_value = torch.where(
                    torch.isinf(hidden_states).any(),
                    torch.finfo(hidden_states.dtype).max - 1000,
                    torch.finfo(hidden_states.dtype).max,
                )
                hidden_states = torch.clamp(
                    hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + \
                    cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        #### NEW CODE ####
        if self.has_downsampling_module:
            outputs = outputs + (attention_mask,)
        #### NEW CODE ####

        return outputs



class CanineT5Stack(T5Stack):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens)

        ##### NEW CODE #####
        if self.is_decoder:
            self.block = nn.ModuleList(
                [
                    CanineT5Block(
                        config, has_relative_attention_bias=bool(i == 0))
                    for i in range(config.num_layers)
                ]
            )
        else:
            blocks = []
            for i in range(config.num_layers):
                blocks.append(
                    CanineT5Block(
                        config,
                        # Only the first layer has relative attention bias
                        has_relative_attention_bias=bool(i == 0),
                        # Add delete gate if specified
                        has_downsampling_module=bool(i == config.downsampling_layer),
                    )
                )
            self.block = nn.ModuleList(blocks)
        ##### NEW CODE #####

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise ValueError(
                    "You have to initialize the model with valid token embeddings")
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + \
            seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            if not self.is_decoder:
                raise ValueError(
                    f"`use_cache` can only be set to `True` if {self} is used as a decoder")

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, mask_seq_length, device=inputs_embeds.device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (
                encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=inputs_embeds.device, dtype=torch.long
                )
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        #### NEW CODE ####
        # Return a new encoder attention mask if hard delete is enabled
        attention_mask_to_return = None
        #### NEW CODE ####

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(
            cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(
                        hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(
                        hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(
                        hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(
                        hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.forward,
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                    use_cache,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            #### NEW CODE ####
            # Get the new attention mask
            if layer_module.has_downsampling_module:
                extended_attention_mask = layer_outputs[-1]
                attention_mask_to_return = extended_attention_mask.squeeze(-2).squeeze(-2)
                attention_mask_to_return = (attention_mask_to_return == 0).int()

            #### NEW CODE ####

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + \
                        (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return CanineT5BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
            #### NEW CODE ####
            attention_mask=attention_mask_to_return,
            #### NEW CODE ####
        )


class CanineT5ForConditionalGeneration(T5ForConditionalGeneration):
    def __init__(self, config: CanineT5Config):
        super().__init__(config)
        #### NEW CODE ####
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = CanineT5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = CanineT5Stack(decoder_config, self.shared)
        #### NEW CODE ####

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            #### NEW CODE ####
            encoder_outputs = CanineT5BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=encoder_outputs.last_hidden_state,
                hidden_states=encoder_outputs.hidden_states if 'hidden_states' in encoder_outputs else None,
                attentions=encoder_outputs.attentions if 'attentions' in encoder_outputs else None,
            )
            #### NEW CODE ####

        #### NEW CODE ####

        hidden_states = encoder_outputs.last_hidden_state
        attention_mask = encoder_outputs.attention_mask
        
        #### NEW CODE ####

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(
                    self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(
                    self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(
                lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        ##### NEW CODE #####
        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
        ##### NEW CODE #####
