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
    T5Config,
    T5ForConditionalGeneration,
    softmax1,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
)
from transformers.utils import logging
from typing import Optional, Tuple, Union
from dataclasses import dataclass

logger = logging.get_logger(__name__)


class MrT5Config(T5Config):
    def __init__(
        self,
        *args,
        sigmoid_mask_scale=-10.0,
        gate_layer_norm=True,
        deletion_threshold=None,
        delete_gate_layer=2,
        use_softmax1=False,
        deletion_type=None,
        random_deletion_probability=0.5,
        fixed_deletion_amount=0.5,
        train_language="en",
        eval_language="en",
        output_attn_logs=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.deletion_threshold = deletion_threshold
        self.sigmoid_mask_scale = sigmoid_mask_scale
        self.gate_layer_norm = gate_layer_norm
        self.use_softmax1 = use_softmax1
        self.deletion_type = deletion_type
        self.random_deletion_probability = random_deletion_probability
        self.fixed_deletion_amount = fixed_deletion_amount
        self.train_language = train_language
        self.eval_language = eval_language
        self.output_attn_logs = output_attn_logs
        self.delete_gate_layer = delete_gate_layer


@dataclass
class MrT5BaseModelOutputWithPastAndCrossAttentions(BaseModelOutputWithPastAndCrossAttentions):
    delete_gate_mask: torch.FloatTensor = None
    delete_gate_output: torch.FloatTensor = None
    delete_gate_logits: torch.FloatTensor = None
    attn_logs: torch.FloatTensor = None
    cross_attention_attn_logs: torch.FloatTensor = None
    attention_mask: torch.FloatTensor = None


@dataclass
class MrT5Seq2SeqLMOutput(Seq2SeqLMOutput):
    delete_gate_mask: torch.FloatTensor = None
    delete_gate_output: torch.FloatTensor = None
    delete_gate_logits: torch.FloatTensor = None
    encoder_attn_logs: torch.FloatTensor = None
    decoder_attn_logs: torch.FloatTensor = None
    cross_attention_attn_logs: torch.FloatTensor = None


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
            self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.feed_forward = nn.Linear(config.hidden_size, 1)
        self._init_weights(self.feed_forward)
        self.activation = ScaledSigmoid(config.sigmoid_mask_scale)

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
            m.bias.data.fill_(-1)


class LogSigmoidDeleteGate(SigmoidDeleteGate):
    def __init__(self, config):
        super().__init__(config)
        self.activation = nn.LogSigmoid()

class RandomDeleteGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Store the sigmoid_mask_scale and the probability of activation
        self.sigmoid_mask_scale = config.sigmoid_mask_scale
        self.random_deletion_probability = config.random_deletion_probability

    def __random_mask_tensor(self, x, n):
        # Determine the shape for the output tensor
        target_shape = (x.shape[0], x.shape[1], 1)
        total_elements = x.shape[0] * x.shape[1]
        
        # Create a flattened float tensor of all 0.0
        flat_tensor = torch.zeros(total_elements, dtype=torch.float32, device=x.device)
        
        # Randomly select n indices to be set to 1.0
        indices = torch.randperm(total_elements)[:n]
        flat_tensor[indices] = 1.0
        
        # Reshape it to match the desired target shape
        float_tensor = flat_tensor.view(target_shape)
        
        return float_tensor

    def forward(self, hidden_states, input_ids):
        # Calculate the number of tokens to delete using a gaussian
        deletion_percentage = np.random.normal(loc=self.random_deletion_probability, scale=0.05)
        n_deletions = int(deletion_percentage * hidden_states.shape[0] * hidden_states.shape[1])
        
        # Create a random mask with n_deletions True values
        random_mask = self.__random_mask_tensor(hidden_states, n_deletions)
        
        # Scale the mask by sigmoid_mask_scale
        delete_gate_mask = random_mask * self.sigmoid_mask_scale
        return delete_gate_mask, delete_gate_mask

    
class FixedDeleteGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sigmoid_mask_scale = config.sigmoid_mask_scale
        self.fixed_deletion_amount = config.fixed_deletion_amount
        self.sep_tokens = torch.tensor([12, 13, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                                        46, 47, 48, 49, 50, 61, 62, 63, 64, 65, 66, 67, 94,
                                        95, 96, 97, 98, 99, 126, 127, 128, 129, 1])

    def __create_mask(self, input_ids):
        device = input_ids.device
        batch_size, seq_len = input_ids.size()
        self.sep_tokens = self.sep_tokens.to(device)
        
        # Create an initial mask filled with sigmoid_mask_scale
        mask = torch.full((batch_size, seq_len), self.sigmoid_mask_scale, device=device)
        
        # Find sep_token indices
        is_sep = torch.isin(input_ids, self.sep_tokens)

        # Create a tensor of segment lengths
        sep_positions = torch.cumsum(is_sep, dim=1)
        segment_lengths = torch.zeros_like(input_ids, dtype=torch.float)
        segment_lengths[:, 1:] = (sep_positions[:, 1:] != sep_positions[:, :-1]).float()
        segment_lengths[:, 0] = 1.0
        segment_lengths = torch.cumsum(segment_lengths, dim=1)
        
        # Calculate number of zeros for each segment
        segment_counts = torch.bincount(sep_positions.view(-1), minlength=seq_len)
        segment_starts = torch.cumsum(torch.cat([torch.tensor([0], device=device), segment_counts[:-1]]), dim=0)
        segment_ends = torch.cumsum(segment_counts, dim=0)
        num_zeros = torch.ceil((1 - self.fixed_deletion_amount) * (segment_ends - segment_starts)).long()
        
        # Create the mask based on the calculated number of zeros
        for i in range(batch_size):
            for start, count in zip(segment_starts, num_zeros):
                mask[i, start:start + count] = 0
        
        return mask.to(torch.float)

    def forward(self, hidden_states, input_ids):
        delete_gate_mask = self.__create_mask(input_ids).unsqueeze(-1)
        return delete_gate_mask, delete_gate_mask


class MrT5Attention(T5Attention):
    """
    Extends the T5Attention class to include a delete gate. Only the forward
    method is modified. The delete_gate_mask passed to the forward function
    is applied to the attention scores.
    """

    def __init__(self, config: MrT5Config, has_relative_attention_bias=False):
        super().__init__(config, has_relative_attention_bias)
        #### NEW CODE ####
        self.use_softmax1 = config.use_softmax1 
        #### NEW CODE ####

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
        output_attn_logs=False,
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

        #### NEW CODE ####
        if not self.has_absolute_position_embeddings:
        #### NEW CODE ####
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

            scores += position_bias_masked
        #### NEW CODE ####
        # If there is no position bias, add attention mask to scores directly
        elif mask is not None:
            scores += mask
        
        # Create attention logs, log attention scores before applying gate mask
        attn_logs = (scores.detach().clone(),)

        # Apply the mask from the delete gate
        if delete_gate_mask is not None:
            scores += delete_gate_mask.squeeze(-1).unsqueeze(-2).unsqueeze(-2)

        # Log attention scores after applying gate mask
        attn_logs = attn_logs + (scores,)

        if self.use_softmax1:
            attn_weights = softmax1(scores.float(), dim=-1).type_as(
                scores)
        else:
            attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
                scores
            )  # (batch_size, n_heads, seq_length, key_length)

        # Log attention weights
        attn_logs = attn_logs + (attn_weights,)
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
            outputs = outputs + (attn_weights,)

        #### NEW CODE ####
        if output_attn_logs:
            value_norms = torch.norm(value_states, p=2, dim=3)
            attn_logs = attn_logs + (value_norms,)
            outputs = outputs + (attn_logs,)
        #### NEW CODE ####

        return outputs


class MrT5LayerSelfAttention(nn.Module):
    """
    Modified version of T5LayerSelfAttention that uses MrT5Attention instead
    of T5Attention.
    """

    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        #### NEW CODE ####
        # Use MrT5Attention instead of T5Attention
        self.SelfAttention = MrT5Attention(
            config, has_relative_attention_bias=has_relative_attention_bias)
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
        output_attn_logs=False,
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
            output_attn_logs=output_attn_logs,
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

    def __init__(self, config):
        super().__init__()
        #### NEW CODE ####
        # Use MrT5Attention instead of T5Attention
        self.EncDecAttention = MrT5Attention(
            config, has_relative_attention_bias=False)
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
        output_attn_logs=False,
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
            output_attn_logs=output_attn_logs,
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

    def __init__(self, config, has_relative_attention_bias=False,
                 #### NEW CODE ####
                 has_delete_gate=False,
                 hard_delete_block=False,
                 #### NEW CODE ####
                 ):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        #### NEW CODE ####
        # Use MrT5LayerSelfAttention and MrT5LayerCrossAttention
        # instead of T5LayerSelfAttention and T5LayerCrossAttention
        self.layer.append(MrT5LayerSelfAttention(
            config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(MrT5LayerCrossAttention(config))
        #### NEW CODE ####

        self.layer.append(T5LayerFF(config))

        #### NEW CODE ####
        # Add delete gate if needed
        self.has_delete_gate = has_delete_gate
        if self.has_delete_gate:
            if config.deletion_type == "scaled_sigmoid":
                self.delete_gate = SigmoidDeleteGate(config)
            elif config.deletion_type == "log_sigmoid":
                self.delete_gate = LogSigmoidDeleteGate(config)
            elif config.deletion_type == "random":
                self.delete_gate = RandomDeleteGate(config)
            elif config.deletion_type == "fixed":
                self.delete_gate = FixedDeleteGate(config)
            else:
                raise ValueError(
                    f"Invalid deletion type: {config.deletion_type}")

        # Set hard_delete flags
        self.hard_delete_block = hard_delete_block
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
        output_attn_logs=False,
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
        if self.has_delete_gate:
            delete_gate_mask, delete_gate_logits = self.delete_gate(
                hidden_states, input_ids)

        # Apply hard deletion
        if self.hard_delete_block and hard_delete:

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
            output_attn_logs=output_attn_logs,
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
                output_attn_logs=output_attn_logs,
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
            delete_gate_mask, delete_gate_logits = self.delete_gate(
                hidden_states, input_ids)
            outputs = outputs + (delete_gate_mask, delete_gate_logits)

        # Return new resized masks if hard delete is enabled
        elif self.hard_delete_block and hard_delete:
            outputs = outputs + (delete_gate_mask, attention_mask)
        ##### NEW CODE #####

        # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights), (delete_gate_mask), (delete_gate_logits)
        return outputs


class MrT5Stack(T5Stack):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens)

        ##### NEW CODE #####
        if self.is_decoder:
            self.block = nn.ModuleList(
                [
                    MrT5Block(
                        config, has_relative_attention_bias=bool(i == 0))
                    for i in range(config.num_layers)
                ]
            )
        else:
            blocks = []
            for i in range(config.num_layers):
                blocks.append(
                    MrT5Block(
                        config,
                        # Only the first layer has relative attention bias
                        has_relative_attention_bias=bool(i == 0),
                        # Add delete gate if specified
                        has_delete_gate=bool(i == config.delete_gate_layer),
                        # Add hard delete if previous layer had delete gate
                        hard_delete_block=bool(i-1 == config.delete_gate_layer),
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
        #### NEW CODE ####
        delete_gate_mask=None,
        delete_gate_output=None,
        delete_gate_logits=None,
        hard_delete=None,
        deletion_threshold=None,
        output_attn_logs=None,
        #### NEW CODE ####
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_attn_logs = output_attn_logs if output_attn_logs is not None else self.config.output_attn_logs
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

        #### NEW CODE ####
        if self.absolute_pos_embed is not None:
            position_ids = torch.arange(input_shape[-1], dtype=torch.long, device=inputs_embeds.device)
            position_embeds = self.absolute_pos_embed(position_ids)
            inputs_embeds = inputs_embeds + position_embeds
        #### NEW CODE ####

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
        all_attn_logs = () if output_attn_logs else None
        all_cross_attention_attn_logs = () if (output_attn_logs and self.is_decoder) else None
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
                    #### NEW CODE ####
                    delete_gate_mask,
                    output_attn_logs,
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
                    output_attn_logs=output_attn_logs,
                    #### NEW CODE ####
                )

            #### NEW CODE ####
            # Update delete_gate_mask if the previous layer had a delete gate
            if layer_module.has_delete_gate:
                delete_gate_mask, delete_gate_logits = layer_outputs[-2], layer_outputs[-1]
                delete_gate_output = delete_gate_mask

            # Update resized masks if the previous layer did a hard deletion
            if layer_module.hard_delete_block and hard_delete:
                delete_gate_mask, extended_attention_mask = layer_outputs[-2], layer_outputs[-1]
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
                index += 1 if output_attn_logs else 0
                encoder_decoder_position_bias = layer_outputs[index]
                #### NEW CODE ####
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + \
                    (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + \
                        (layer_outputs[5],)
                    
            #### NEW CODE ####
            if output_attn_logs:
                all_attn_logs = all_attn_logs + (layer_outputs[-1],)
                if self.is_decoder:
                    all_cross_attention_attn_logs = all_cross_attention_attn_logs + (layer_outputs[-2],)
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
            attn_logs=all_attn_logs,
            cross_attention_attn_logs=all_cross_attention_attn_logs,
            attention_mask=attention_mask_to_return,
            #### NEW CODE ####
        )


class MrT5ForConditionalGeneration(T5ForConditionalGeneration):
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
        output_attn_logs: Optional[bool] = None,
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
                output_attn_logs=output_attn_logs,
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
            output_attn_logs=output_attn_logs,
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
            encoder_attn_logs=encoder_outputs.attn_logs,
            decoder_attn_logs=decoder_outputs.attn_logs,
            cross_attention_attn_logs=decoder_outputs.cross_attention_attn_logs,
        )
        ##### NEW CODE #####
