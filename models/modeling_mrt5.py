# modeling_mrt5.py
# Author: Julie Kallini
# Description: This file contains the implementation of the MrT5 model.
# The code is adapted from HuggingFace's modeling_t5.py. New code sequences
# are labeled with comments.

import torch
import copy
import numpy as np
from torch import nn
from models.modeling_t5 import (
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
from transformers.utils import logging
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

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
        #### NEW CODE ####
        delete_gate_mask=None,
        #### NEW CODE ####
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            if len(past_key_value) != 2:
                raise ValueError(
                    f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
                )
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[
            1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat(
                        [past_key_value, hidden_states], dim=2)
                elif past_key_value.shape[2] != key_value_states.shape[1]:
                    # checking that the `sequence_length` of the `past_key_value` is the same as
                    # the provided `key_value_states` to support prefix tuning
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        # (batch_size, n_heads, seq_length, dim_per_head)
        query_states = shape(self.q(hidden_states))

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[
                0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[
                1] if past_key_value is not None else None
        )

        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(
                    real_seq_length, key_length, device=scores.device)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1):, :]

            if mask is not None:
                # (batch_size, n_heads, seq_length, key_length)
                position_bias = position_bias + mask

        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        scores = scores + position_bias_masked

        #### NEW CODE ####
        # Log scores to return for loss calculation
        scores_to_return = scores

        # Apply the mask from the delete gate
        if delete_gate_mask is not None:
            scores = scores + delete_gate_mask.squeeze(-1).unsqueeze(-2).unsqueeze(-2)

        attn_weights = softmax1(scores.float(), dim=-1).type_as(scores)
        #### NEW CODE ####

        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        # (batch_size, seq_length, dim)
        attn_output = unshape(torch.matmul(attn_weights, value_states))
        attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if (
            self.is_decoder and use_cache) else None
        outputs = (attn_output,) + \
            (present_key_value_state,) + (position_bias,)

        if output_attentions:
            attentions_keys_queries = (attn_weights, key_states, query_states, value_states, scores_to_return)
            outputs = outputs + (attentions_keys_queries,)

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

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
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
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            #### NEW CODE ####
            delete_gate_mask=delete_gate_mask,
            #### NEW CODE ####
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        # add attentions if we output them
        outputs = (hidden_states,) + attention_output[1:]
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

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
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
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
            #### NEW CODE ####
            delete_gate_mask=delete_gate_mask,
            #### NEW CODE ####
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        # add attentions if we output them
        outputs = (layer_output,) + attention_output[1:]
        return outputs


class MrT5Block(nn.Module):
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
        delete_gate_mask=None,
        input_ids=None,
        hard_delete=None,
        deletion_threshold=None,
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
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            #### NEW CODE ####
            # Only apply delete_gate_mask to self-attention if the block
            # is the encoder
            delete_gate_mask=None if self.is_decoder else delete_gate_mask,
            #### NEW CODE ####
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

        ##### NEW CODE #####
        if self.has_delete_gate:
            outputs = outputs + \
                (delete_gate_values, delete_gate_logits, delete_gate_mask, attention_mask)
        ##### NEW CODE #####

        # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights), (delete_gate_mask), (delete_gate_logits)
        return outputs


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
                    #### NEW CODE ####
                    delete_gate_mask,
                    #### NEW CODE ####
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
                    #### NEW CODE ####
                    delete_gate_mask=delete_gate_mask,
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
                #### NEW CODE ####
                index = 4 if output_attentions else 3
                encoder_decoder_position_bias = layer_outputs[index]
                #### NEW CODE ####
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + \
                    (present_key_value_state,)

            #### NEW CODE ####
            if output_attentions:
                attn_weights, keys, queries, values, scores = layer_outputs[3]
                all_attentions = all_attentions + (attn_weights,)
                all_queries = all_queries + (queries,)
                all_keys = all_keys + (keys,)
                all_values = all_values + (values,)
                all_scores = all_scores + (scores,)

                if self.is_decoder:
                    cross_attn_weights, cross_attn_keys, cross_attn_queries, \
                        cross_attn_values, cross_attn_scores = layer_outputs[5]
                    all_cross_attentions = all_cross_attentions + \
                        (cross_attn_weights,)
                    all_cross_attn_queries = all_cross_attn_queries + \
                        (cross_attn_queries,)
                    all_cross_attn_keys = all_cross_attn_keys + \
                        (cross_attn_keys,)
                    all_cross_attn_values = all_cross_attn_values + \
                        (cross_attn_values,)
                    all_cross_attn_scores = all_cross_attn_scores + \
                        (cross_attn_scores,)
            #### NEW CODE ####

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
                    #### NEW CODE ####
                    delete_gate_mask,
                    delete_gate_output,
                    delete_gate_logits,
                    attention_mask_to_return,
                    all_queries,
                    all_keys,
                    all_values,
                    all_scores,
                    all_cross_attn_queries,
                    all_cross_attn_keys,
                    all_cross_attn_values,
                    all_cross_attn_scores,
                    #### NEW CODE ####
                ]
                if v is not None
            )

        return MrT5BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
            #### NEW CODE ####
            delete_gate_mask=delete_gate_mask,
            delete_gate_output=delete_gate_output,
            delete_gate_logits=delete_gate_logits,
            attention_mask=attention_mask_to_return,
            attention_queries=all_queries,
            attention_keys=all_keys,
            attention_values=all_values,
            attention_scores=all_scores,
            cross_attention_queries=all_cross_attn_queries,
            cross_attention_keys=all_cross_attn_keys,
            cross_attention_values=all_cross_attn_values,
            cross_attention_scores=all_cross_attn_scores,
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
        encoder_config.is_encoder_decoder = False
        self.encoder = MrT5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
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

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        ##### NEW CODE #####
        # TODO: Generation will need special handling of attention masks, which
        # will need to be resized if hard delete is enabled. For now, we will
        # simply omit the encoder attention mask for generation.
        attention_mask = None
        ##### NEW CODE #####

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }
