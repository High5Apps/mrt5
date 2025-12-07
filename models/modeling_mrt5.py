# modeling_mrt5.py
# Author: Julie Kallini
# Description: This file contains the implementation of the MrT5 model.
# The code is adapted from HuggingFace's modeling_t5.py. New code sequences
# are labeled with comments.

import torch
import copy
from torch import nn
from transformers import GradientCheckpointingLayer
from transformers.cache_utils import DynamicCache, EncoderDecoderCache
from transformers.models.t5.modeling_t5 import (
    T5Attention,
    T5LayerNorm,
    T5LayerFF,
    T5Stack,
    T5ForConditionalGeneration,
)
from .configuration_mrt5 import MrT5Config
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
)
from transformers.utils import logging, is_torchdynamo_compiling
from transformers.utils.deprecation import deprecate_kwarg
from typing import Optional, Tuple, Union
from dataclasses import dataclass

logger = logging.get_logger(__name__)

@dataclass
class MrT5BaseModelOutputWithPastAndCrossAttentions(BaseModelOutputWithPastAndCrossAttentions):
    delete_gate_mask: torch.FloatTensor = None
    delete_gate_output: torch.FloatTensor = None
    delete_gate_logits: torch.FloatTensor = None
    attention_mask: torch.FloatTensor = None
    attention_queries: torch.FloatTensor = None
    attention_keys: torch.FloatTensor = None
    attention_values: torch.FloatTensor = None
    attention_scores: torch.FloatTensor = None
    cross_attention_keys: torch.FloatTensor = None
    cross_attention_queries: torch.FloatTensor = None
    cross_attention_values: torch.FloatTensor = None
    cross_attention_scores: torch.FloatTensor = None


@dataclass
class MrT5Seq2SeqLMOutput(Seq2SeqLMOutput):
    delete_gate_mask: torch.FloatTensor = None
    delete_gate_output: torch.FloatTensor = None
    delete_gate_logits: torch.FloatTensor = None
    encoder_keys: torch.FloatTensor = None
    encoder_queries: torch.FloatTensor = None
    encoder_values: torch.FloatTensor = None
    encoder_scores: torch.FloatTensor = None
    decoder_keys: torch.FloatTensor = None
    decoder_queries: torch.FloatTensor = None
    decoder_values: torch.FloatTensor = None
    decoder_scores: torch.FloatTensor = None
    cross_attention_keys: torch.FloatTensor = None
    cross_attention_queries: torch.FloatTensor = None
    cross_attention_values: torch.FloatTensor = None
    cross_attention_scores: torch.FloatTensor = None


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

def softmax1(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if logits.shape[dim] == 0:
        return logits
    m = logits.detach().max(dim, keepdim=True)[0]
    logits = logits - m
    logits = logits.exp()
    return logits / (logits.sum(dim, keepdim=True) + (-m).exp())

class ScaledSigmoid(nn.Module):
    def __init__(self, sigmoid_mask_scale):
        super().__init__()
        self.sigmoid_mask_scale = sigmoid_mask_scale

    def forward(self, input):
        return self.sigmoid_mask_scale * torch.sigmoid(-input)

class SigmoidDeleteGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.has_layer_norm = config.gate_layer_norm
        if self.has_layer_norm:
            self.layer_norm = T5LayerNorm(config.hidden_size)
        self.feed_forward = nn.Linear(config.hidden_size, 1)
        self._init_weights(self.feed_forward)
        self.activation = ScaledSigmoid(config.sigmoid_mask_scale)
        self.use_gumbel_noise = config.use_gumbel_noise

    def forward(self, hidden_states, input_ids):
        if self.has_layer_norm:
            hidden_states = self.layer_norm(hidden_states)
        delete_gate_logits = self.feed_forward(hidden_states)

        gate_values = self.activation(delete_gate_logits)

        # Check if there are any pad tokens in input_ids
        if (input_ids == 0).any():
            # Set gate values for pad tokens (input_ids == 0) to sigmoid_mask_scale
            pad_mask = (input_ids == 0).unsqueeze(-1)
            gate_values = torch.where(pad_mask, torch.tensor(self.activation.sigmoid_mask_scale), gate_values)

        return gate_values, delete_gate_logits

    def _init_weights(self, m, init_func="xavier_uniform_"):
        # Initialize the weights. This is necessary because
        # HuggingFace disables initialization during "from_pretrained"
        if isinstance(m, nn.Linear):
            TORCH_INIT_FUNCTIONS[init_func](m.weight)
            m.bias.data.fill_(1)

class MrT5Attention(T5Attention):
    """
    Extends the T5Attention class to include a delete gate. Only the forward
    method is modified. The delete_gate_mask passed to the forward function
    is applied to the attention scores.
    """

    def __init__(self, config: MrT5Config, has_relative_attention_bias=False, layer_idx: Optional[int] = None):
        super().__init__(config, has_relative_attention_bias, layer_idx)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_values=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
        cache_position=None,
        #### NEW CODE ####
        delete_gate_mask=None,
        #### NEW CODE ####
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, 1, 1, key_length) (non-causal encoder) or (batch_size, 1, seq_length, key_length) (causal decoder)
        batch_size, seq_length = hidden_states.shape[:2]

        # if key_value_states are provided this layer is used as a cross-attention layer for the decoder
        is_cross_attention = key_value_states is not None

        query_states = self.q(hidden_states)
        query_states = query_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        # Check is encoder-decoder model is being used. Otherwise we'll get `DynamicCache`
        is_updated = False
        if isinstance(past_key_values, EncoderDecoderCache):
            is_updated = past_key_values.is_updated.get(self.layer_idx)
            if is_cross_attention:
                # after the first generated id, we can subsequently re-use all key/value_states from cache
                curr_past_key_value = past_key_values.cross_attention_cache
            else:
                curr_past_key_value = past_key_values.self_attention_cache
        else:
            curr_past_key_value = past_key_values

        current_states = key_value_states if is_cross_attention else hidden_states
        if is_cross_attention and past_key_values is not None and is_updated:
            # reuse k,v, cross_attentions
            key_states = curr_past_key_value.layers[self.layer_idx].keys
            value_states = curr_past_key_value.layers[self.layer_idx].values
        else:
            key_states = self.k(current_states)
            value_states = self.v(current_states)
            key_states = key_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
            value_states = value_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

            if past_key_values is not None:
                # save all key/value_states to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = curr_past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )
                # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
                if is_cross_attention and isinstance(past_key_values, EncoderDecoderCache):
                    past_key_values.is_updated[self.layer_idx] = True

        # compute scores, equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9
        scores = torch.matmul(query_states, key_states.transpose(3, 2))

        if position_bias is None:
            key_length = key_states.shape[-2]
            # cache position is 0-indexed so we add 1 to get the real length of queries (aka with past)
            real_seq_length = query_length if query_length is not None else cache_position[-1] + 1
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(
                    real_seq_length, key_length, device=scores.device, cache_position=cache_position
                )
                position_bias = position_bias[:, :, -seq_length:, :]

            if mask is not None:
                causal_mask = mask[:, :, :, : key_states.shape[-2]]
                position_bias = position_bias + causal_mask

        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        scores += position_bias_masked

        #### NEW CODE ####

        # Apply the mask from the delete gate
        if delete_gate_mask is not None:
            scores = scores + delete_gate_mask.squeeze(-1).unsqueeze(-2).unsqueeze(-2)

        attn_weights = softmax1(scores.float(), dim=-1).type_as(scores)
        #### NEW CODE ####

        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.inner_dim)
        attn_output = self.o(attn_output)

        outputs = (attn_output, position_bias)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class MrT5LayerSelfAttention(nn.Module):
    """
    Modified version of T5LayerSelfAttention that uses MrT5Attention instead
    of T5Attention.
    """

    def __init__(self, config, has_relative_attention_bias=False, layer_idx: Optional[int] = None):
        super().__init__()
        #### NEW CODE ####
        # Use MrT5Attention instead of T5Attention
        self.SelfAttention = MrT5Attention(
            config, has_relative_attention_bias=has_relative_attention_bias, layer_idx=layer_idx)
        #### NEW CODE ####
        self.layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_values=None,
        use_cache=False,
        output_attentions=False,
        cache_position=None,
        #### NEW CODE ####
        delete_gate_mask=None,
        #### NEW CODE ####
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
            #### NEW CODE ####
            delete_gate_mask=delete_gate_mask,
            #### NEW CODE ####
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


class MrT5LayerCrossAttention(nn.Module):
    """
    Modified version of T5LayerCrossAttention that uses MrT5Attention instead
    of T5Attention.
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        #### NEW CODE ####
        # Use MrT5Attention instead of T5Attention
        self.EncDecAttention = MrT5Attention(
            config, has_relative_attention_bias=False, layer_idx=layer_idx)
        #### NEW CODE ####
        self.layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_values=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
        cache_position=None,
        #### NEW CODE ####
        delete_gate_mask=None,
        #### NEW CODE ####
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
            cache_position=cache_position,
            #### NEW CODE ####
            delete_gate_mask=delete_gate_mask,
            #### NEW CODE ####
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


class MrT5Block(GradientCheckpointingLayer):
    """
    Modified version of T5Block that uses MrT5LayerSelfAttention and
    MrT5LayerCrossAttention instead of T5LayerSelfAttention and
    T5LayerCrossAttention.
    """

    def __init__(self, config, has_relative_attention_bias=False, layer_idx: Optional[int] = None):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        #### NEW CODE ####
        # Use MrT5LayerSelfAttention and MrT5LayerCrossAttention
        # instead of T5LayerSelfAttention and T5LayerCrossAttention
        self.layer.append(MrT5LayerSelfAttention(
            config, has_relative_attention_bias=has_relative_attention_bias, layer_idx=layer_idx))
        if self.is_decoder:
            self.layer.append(MrT5LayerCrossAttention(config, layer_idx=layer_idx))
        #### NEW CODE ####

        self.layer.append(T5LayerFF(config))

        #### NEW CODE ####
        # Add delete gate if needed
        self.has_delete_gate = not config.is_decoder and (layer_idx == config.delete_gate_layer)
        if self.has_delete_gate:
            if config.deletion_type == "scaled_sigmoid":
                self.delete_gate = SigmoidDeleteGate(config)
            else:
                raise ValueError(
                    f"Invalid deletion type: {config.deletion_type}")

        # Set hard_delete flags
        self.sigmoid_mask_scale = config.sigmoid_mask_scale
        self.deletion_threshold = config.deletion_threshold
        #### NEW CODE ####

    #### NEW CODE ####
    
    def __get_new_positions_and_mask(self, batch_size, seq_len, delete_gate_mask, deletion_threshold, device):
        delete_gate_mask = delete_gate_mask.squeeze(-1)

        # Create filter from delete gate mask
        deletion_threshold = deletion_threshold if deletion_threshold is not None else self.deletion_threshold
        keep_this = delete_gate_mask > deletion_threshold

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

        # Create the new mask
        new_mask = torch.arange(new_len, device=device).expand(batch_size, -1) <= target_pos[:, -1:]
        new_mask = (~new_mask).float() * -1e9
        new_mask = new_mask.unsqueeze(-1)

        return src_side_pos.long(), new_mask
    
    def __hard_delete_hidden_states(self, hidden_states, positions):
        new_hidden_states = torch.gather(hidden_states, 1, positions.unsqueeze(2).expand(-1, -1, hidden_states.size(2)))
        return new_hidden_states
    
    def __hard_delete_4_dimensions(self, position_bias, positions):
        new_position_bias = torch.gather(position_bias, 1, positions.unsqueeze(2).unsqueeze(3).expand(-1, -1, position_bias.size(2), position_bias.size(3)))
        return new_position_bias
    
    #### NEW CODE ####

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        #### NEW CODE ####
        delete_gate_mask=None,
        #### NEW CODE ####
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_values=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
        cache_position=None,
        #### NEW CODE ####
        input_ids=None,
        hard_delete=None,
        deletion_threshold=None,
        #### NEW CODE ####
    ):
        ##### NEW CODE #####
        # Initialize delete gate values and logits for logging/loss calculation
        delete_gate_values = None
        delete_gate_logits = None

        if self.has_delete_gate:
            delete_gate_values, delete_gate_logits = self.delete_gate(
                hidden_states, input_ids)
            delete_gate_mask = delete_gate_values

            # Raise error if all tokens are deleted in any sequence in batch
            if (delete_gate_values < self.deletion_threshold).all():
                raise ValueError("All tokens are deleted in this batch. " + \
                                 "Please adjust the deletion rate or " + \
                                 "alpha hyperparameter.")

            # Apply hard deletion
            if hard_delete:

                # Compute new token positions
                new_positions, delete_gate_mask = self.__get_new_positions_and_mask(
                    hidden_states.size(0), hidden_states.size(1), delete_gate_mask, deletion_threshold, hidden_states.device)

                # Compute new position bias
                if position_bias is not None:
                    new_position_bias = self.__hard_delete_4_dimensions(
                        position_bias.permute(0, 2, 3, 1), new_positions)
                    new_position_bias = self.__hard_delete_4_dimensions(
                        new_position_bias.permute(0, 2, 1, 3), new_positions)
                    position_bias = new_position_bias.permute(0, 3, 2, 1)

                # Compute new attention mask
                new_attention_mask = self.__hard_delete_4_dimensions(
                    attention_mask.permute(0, 3, 1, 2), new_positions)
                attention_mask = new_attention_mask.permute(0, 2, 3, 1)

                # Compute new hidden states and delete gate mask
                hidden_states = self.__hard_delete_hidden_states(
                    hidden_states, new_positions)

        ##### NEW CODE #####

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
            #### NEW CODE ####
            # Only apply delete_gate_mask to self-attention if the block
            # is the encoder
            delete_gate_mask=None if self.is_decoder else delete_gate_mask,
            #### NEW CODE ####
        )
        hidden_states = self_attention_outputs[0]
        attention_outputs = self_attention_outputs[1:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_values=past_key_values,
                query_length=cache_position[-1] + 1,
                use_cache=use_cache,
                output_attentions=output_attentions,
                #### NEW CODE ####
                delete_gate_mask=delete_gate_mask,
                #### NEW CODE ####
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16:
                clamp_value = torch.where(
                    torch.isinf(hidden_states).any(),
                    torch.finfo(hidden_states.dtype).max - 1000,
                    torch.finfo(hidden_states.dtype).max,
                )
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[1:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)
        ##### NEW CODE #####
        # hidden-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
        outputs = outputs + attention_outputs

        if self.has_delete_gate:
            outputs = outputs + \
                (delete_gate_values, delete_gate_logits, delete_gate_mask, attention_mask)
        
        # Tigler note: it seems like these don't match up with the comment directly above
        # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights), (delete_gate_mask), (delete_gate_logits)
        return outputs
        ##### NEW CODE #####


class MrT5Stack(T5Stack):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens)

        ##### NEW CODE #####
        self.block = nn.ModuleList(
            [MrT5Block(config, has_relative_attention_bias=bool(i == 0), layer_idx=i) for i in range(config.num_layers)]
        )
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
        cache_position=None,
        #### NEW CODE ####
        delete_gate_mask=None,
        delete_gate_output=None,
        delete_gate_logits=None,
        hard_delete=None,
        deletion_threshold=None,
        #### NEW CODE ####
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
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise ValueError("You have to initialize the model with valid token embeddings")
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        if use_cache is True:
            if not self.is_decoder:
                raise ValueError(f"`use_cache` can only be set to `True` if {self} is used as a decoder")

        if self.is_decoder:
            if use_cache and past_key_values is None:
                if self.config.is_encoder_decoder:
                    past_key_values = EncoderDecoderCache(
                        DynamicCache(config=self.config), DynamicCache(config=self.config)
                    )
                else:
                    past_key_values = DynamicCache(config=self.config)
        elif not self.is_decoder:
            # do not pass cache object down the line for encoder stack
            # it messes indexing later in decoder-stack because cache object is modified in-place
            past_key_values = None

        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(
                past_key_values_length, past_key_values_length + seq_length, device=inputs_embeds.device
            )

        if attention_mask is None and not is_torchdynamo_compiling():
            # required mask seq length can be calculated via length of past cache
            mask_seq_length = past_key_values_length + seq_length
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)

        if self.config.is_decoder:
            causal_mask = self._update_causal_mask(
                attention_mask,
                inputs_embeds,
                cache_position,
                past_key_values.self_attention_cache
                if isinstance(past_key_values, EncoderDecoderCache)
                else past_key_values,
                output_attentions,
            )
        elif attention_mask is not None:
            causal_mask = attention_mask[:, None, None, :]
            causal_mask = causal_mask.to(dtype=inputs_embeds.dtype)
            causal_mask = (1.0 - causal_mask) * torch.finfo(inputs_embeds.dtype).min
        else:
            causal_mask = None

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=inputs_embeds.device, dtype=torch.long
                )
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        #### NEW CODE ####
        # Return a new encoder attention mask if hard delete is enabled
        attention_mask_to_return = None
        #### NEW CODE ####

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        #### NEW CODE ####
        all_queries = () if output_attentions else None
        all_keys = () if output_attentions else None
        all_values = () if output_attentions else None
        all_scores = () if output_attentions else None
        all_cross_attn_queries = () if (output_attentions and self.is_decoder) else None
        all_cross_attn_keys = () if (output_attentions and self.is_decoder) else None
        all_cross_attn_values = () if (output_attentions and self.is_decoder) else None
        all_cross_attn_scores = () if (output_attentions and self.is_decoder) else None
        #### NEW CODE ####

        hidden_states = self.dropout(inputs_embeds)

        for i, layer_module in enumerate(self.block):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if causal_mask is not None:
                    causal_mask = causal_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                causal_mask,
                position_bias,
                encoder_hidden_states,
                encoder_extended_attention_mask,
                encoder_decoder_position_bias,  # as a positional argument for gradient checkpointing
                #### NEW CODE ####
                delete_gate_mask,
                #### NEW CODE ####
                layer_head_mask=layer_head_mask,
                cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                return_dict=return_dict,
                cache_position=cache_position,
                #### NEW CODE ####
                input_ids=input_ids,
                hard_delete=hard_delete,
                deletion_threshold=deletion_threshold,
                #### NEW CODE ####
            )

            #### NEW CODE ####
            # Update delete_gate_mask if the previous layer had a delete gate
            if layer_module.has_delete_gate:
                delete_gate_output, delete_gate_logits, delete_gate_mask, new_attention_mask = layer_outputs[-4], layer_outputs[-3], layer_outputs[-2], layer_outputs[-1]

                # Update resized masks if the previous layer did a hard deletion
                if hard_delete:
                    extended_attention_mask = new_attention_mask
                    attention_mask_to_return = extended_attention_mask.squeeze(-2).squeeze(-2)
                    attention_mask_to_return = (attention_mask_to_return == 0).int()

            #### NEW CODE ####

            hidden_states = layer_outputs[0]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[1]
            if self.is_decoder and encoder_hidden_states is not None:
                #### NEW CODE ####
                # Tigler note: This is 1 less than what it was before (4 and 3)
                encoder_decoder_position_bias = layer_outputs[3 if output_attentions else 2]
                #### NEW CODE ####

            # Tigler note: I didn't add the new code here or in the self.is_decoder block below
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[4],)

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
                    past_key_values,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                    #### NEW CODE ####
                    delete_gate_mask,
                    delete_gate_output,
                    delete_gate_logits,
                    attention_mask_to_return,
                    # Tigler note: I didn't include some of these based on the note above
                    #### NEW CODE ####
                ]
                if v is not None
            )
        return MrT5BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
            #### NEW CODE ####
            delete_gate_mask=delete_gate_mask,
            delete_gate_output=delete_gate_output,
            delete_gate_logits=delete_gate_logits,
            attention_mask=attention_mask_to_return,
            # Tigler note: Some of these weren't included for the same reason as the previous note
            #### NEW CODE ####
        )


class MrT5ForConditionalGeneration(T5ForConditionalGeneration):
    
    config_class = MrT5Config

    def __init__(self, config: MrT5Config):
        super().__init__(config)
        #### NEW CODE ####
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        self.encoder = MrT5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = MrT5Stack(decoder_config, self.shared)
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
        cache_position: Optional[torch.LongTensor] = None,
        #### NEW CODE ####
        hard_delete: bool = False,
        deletion_threshold: Optional[float] = None,
        #### NEW CODE ####
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
                #### NEW CODE ####
                hard_delete=hard_delete,
                deletion_threshold=deletion_threshold,
                #### NEW CODE ####
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            #### NEW CODE ####
            encoder_outputs = MrT5BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=encoder_outputs.last_hidden_state,
                hidden_states=encoder_outputs.hidden_states if 'hidden_states' in encoder_outputs else None,
                attentions=encoder_outputs.attentions if 'attentions' in encoder_outputs else None,
                delete_gate_mask=encoder_outputs.delete_gate_mask if 'delete_gate_mask' in encoder_outputs else None,
            )
            #### NEW CODE ####

        #### NEW CODE ####
        
        hidden_states = encoder_outputs.last_hidden_state
        attention_mask = encoder_outputs.attention_mask if 'attention_mask' in encoder_outputs else attention_mask
        
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
            cache_position=cache_position,
            #### NEW CODE ####
            delete_gate_mask=encoder_outputs.delete_gate_mask,
            #### NEW CODE ####
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
        return MrT5Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            delete_gate_mask=encoder_outputs.delete_gate_mask,
            delete_gate_output=encoder_outputs.delete_gate_output,
            delete_gate_logits=encoder_outputs.delete_gate_logits,
            encoder_keys=encoder_outputs.attention_keys,
            encoder_queries=encoder_outputs.attention_queries,
            encoder_values=encoder_outputs.attention_values,
            encoder_scores=encoder_outputs.attention_scores,
            decoder_keys=decoder_outputs.attention_keys,
            decoder_queries=decoder_outputs.attention_queries,
            decoder_values=decoder_outputs.attention_values,
            decoder_scores=decoder_outputs.attention_scores,
            cross_attention_queries=decoder_outputs.cross_attention_queries,
            cross_attention_keys=decoder_outputs.cross_attention_keys,
            cross_attention_values=decoder_outputs.cross_attention_values,
            cross_attention_scores=decoder_outputs.cross_attention_scores,
        )
        ##### NEW CODE #####
