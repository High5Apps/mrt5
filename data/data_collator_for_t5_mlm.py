# data_collator_for_t5_mlm.py
# Author: Julie Kallini
#
# This file is a modified version of the original file from the Hugging Face Transformers library.
# The original file can be found at: https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py
# The original file is licensed under the Apache 2.0 license.
#
# The HuggingFace repository and Google's T5 repository do not provide
# training scripts that are compatible with PyTorch. This file adapts the
# HuggingFace T5 MLM data collator to work with PyTorch. The original
# file was writted to be compatible with JAX/Flax.
#
# This file has also been adapted to be compatible with ByT5 MLM training,
# which specifies sentinel tokens in a slightly different way. This file is
# currently only compatible with ByT5 MLM training.

import numpy as np
import torch
import itertools
from typing import Dict, List
from transformers import (
    BatchEncoding,
    PreTrainedTokenizerBase,
)
from collections import deque


def test_data_collator(input_ids, labels, sentinel_input_ids=258, sentinel_labels=258):
    """
    Processes input ids and labels by removing sentinel values and merging the remaining elements.

    Parameters:
    - input_ids torch.tensor: Queue of input IDs.
    - labels torch.tensor: Queue of label IDs.
    - sentinel_input_ids (int): Sentinel value to indicate end of an input sequence.
    - sentinel_labels (int): Sentinel value to indicate end of a label sequence.

    Returns:
    - list[int]: Merged list of IDs from input_ids and labels, excluding sentinel values.
    """
    merged = []  # This will hold the final merged list of IDs
    # Convert input IDs to a deque for efficient processing
    input_ids = deque(input_ids.tolist())
    # Convert labels to a deque for efficient processing
    labels = deque(labels.tolist()[1:])

    # Process each input ID
    i = 0
    while input_ids:
        current_id = input_ids.popleft()  # Get the next input ID

        # Check if the current ID is the sentinel for input IDs
        if current_id != sentinel_input_ids:
            merged.append(current_id)  # If not a sentinel, add to merged list
        else:
            # If it is a sentinel, update the sentinel and start processing labels
            sentinel_input_ids -= 1
            sentinel_labels -= 1  # Update the label sentinel
            current_label = labels.popleft()  # Get the next label

            # Continue processing labels until the label sentinel is encountered
            while labels and current_label != sentinel_labels:
                merged.append(current_label)  # Add label to merged list
                current_label = labels.popleft()  # Get the next label

    return merged[:-1]  # Return the merged list of IDs


# Tokenization function
def t5_mlm_tokenize_function(examples, chunk_size, tokenizer):
    tokenized = tokenizer(examples["text"], return_attention_mask=False)
    input_ids = list(itertools.chain.from_iterable(tokenized.input_ids))
    max_start_index = len(input_ids) - chunk_size + 1
    chunks = [input_ids[i:i + chunk_size]
                for i in range(0, max_start_index, chunk_size)]
    return BatchEncoding({"input_ids": chunks})


def compute_input_and_target_lengths(inputs_length, noise_density, mean_noise_span_length):
    """This function is copy of `random_spans_helper` :https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2466.

    Training parameters to avoid padding with random_spans_noise_mask.
    When training a model with random_spans_noise_mask, we would like to set the other
    training hyperparmeters in a way that avoids padding.
    This function helps us compute these hyperparameters.
    We assume that each noise span in the input is replaced by extra_tokens_per_span_inputs sentinel tokens,
    and each non-noise span in the targets is replaced by extra_tokens_per_span_targets sentinel tokens.
    This function tells us the required number of tokens in the raw example (for split_tokens())
    as well as the length of the encoded targets. Note that this function assumes
    the inputs and targets will have EOS appended and includes that in the reported length.

    Args:
        inputs_length: an integer - desired length of the tokenized inputs sequence
        noise_density: a float
        mean_noise_span_length: a float
    Returns:
        tokens_length: length of original text in tokens
        targets_length: an integer - length in tokens of encoded targets sequence
    """

    def _tokens_length_to_inputs_length_targets_length(tokens_length):
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        # inputs contain all nonnoise tokens, sentinels for all noise spans
        # and one EOS token.
        _input_length = num_nonnoise_tokens + num_noise_spans + 1
        _output_length = num_noise_tokens + num_noise_spans + 1
        return _input_length, _output_length

    tokens_length = inputs_length

    while _tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0] <= inputs_length:
        tokens_length += 1

    inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(
        tokens_length)

    # minor hack to get the targets length to be equal to inputs length
    # which is more likely to have been set to a nice round number.
    if noise_density == 0.5 and targets_length > inputs_length:
        tokens_length -= 1
        targets_length -= 1
    return tokens_length, targets_length


class DataCollatorForT5MLM:
    """
    Data collator used for T5 span-masked language modeling.
    It is made sure that after masking the inputs are of length `data_args.max_seq_length` and targets are also of fixed length.
    For more information on how T5 span-masked language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.10683.pdf>`__
    or the `official code for preprocessing <https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py>`__ .

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        noise_density (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input.
        mean_noise_span_length (:obj:`float`):
            The average span length of the masked tokens.
        input_length (:obj:`int`):
            The expected input length after masking.
        target_length (:obj:`int`):
            The expected target length after masking.
        pad_token_id: (:obj:`int`):
            The pad token id of the model
        decoder_start_token_id: (:obj:`int):
            The decoder start token id of the model
    """

    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 noise_density: float,
                 mean_noise_span_length: float,
                 input_length: int,
                 target_length: int,
                 pad_token_id: int,
                 decoder_start_token_id: int,
                 rng: np.random.Generator = None,
                 debug_mode: bool = False,
                 ):
        self.tokenizer = tokenizer
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.input_length = input_length
        self.target_length = target_length
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.rng = rng if rng is not None else np.random.default_rng()
        self.debug_mode = debug_mode

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> BatchEncoding:
        # convert list to dict and tensorize input
        batch = BatchEncoding(
            {k: np.array([examples[i][k] for i in range(len(examples))])
             for k, _ in examples[0].items()}
        )

        expanded_input_ids = batch["input_ids"]
        batch_size, expanded_input_length = expanded_input_ids.shape

        mask_indices = np.asarray([self.random_spans_noise_mask(
            expanded_input_length) for _ in range(batch_size)])
        labels_mask = ~mask_indices

        input_ids_sentinel = self.create_sentinel_ids(
            mask_indices.astype(np.int8))
        labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

        batch["input_ids"] = self.filter_input_ids(
            expanded_input_ids, input_ids_sentinel)
        batch["labels"] = self.filter_input_ids(
            expanded_input_ids, labels_sentinel)

        if batch["input_ids"].shape[-1] != self.input_length:
            raise ValueError(
                f"`input_ids` are incorrectly preprocessed. `input_ids` length is {batch['input_ids'].shape[-1]}, but"
                f" should be {self.input_length}."
            )

        if batch["labels"].shape[-1] != self.target_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {batch['labels'].shape[-1]}, but should be"
                f" {self.target_length}."
            )

        # to check that tokens are correctly preprocessed, one can run `self.tokenizer.batch_decode(input_ids)` and `self.tokenizer.batch_decode(labels)` here...
        batch["decoder_input_ids"] = self.shift_tokens_right(
            batch["labels"], self.pad_token_id, self.decoder_start_token_id
        )

        # Convert to torch
        batch["input_ids"] = torch.from_numpy(batch["input_ids"])
        batch["labels"] = torch.from_numpy(batch["labels"])
        batch["decoder_input_ids"] = torch.from_numpy(
            batch["decoder_input_ids"])

        if self.debug_mode:
            for i in range(batch_size):
                # Check that the input_ids and labels are correctly preprocessed
                assert test_data_collator(
                    batch["input_ids"][i], batch["labels"][i]) == expanded_input_ids[i].tolist()

        return batch

    def create_sentinel_ids(self, mask_indices, first_sentinel=258):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - \
            np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(start_indices != 0, np.cumsum(
            start_indices, axis=-1), start_indices)
        sentinel_ids = np.where(
            sentinel_ids != 0, (first_sentinel + 1 - sentinel_ids), 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >=
                                   0].reshape((batch_size, -1))
        input_ids = np.concatenate(
            [input_ids, np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32)], axis=-1
        )
        return input_ids

    def random_spans_noise_mask(self, length):
        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .

        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.

        Args:
            length: an int32 scalar (length of the incoming token sequence)
            noise_density: a float - approximate density of output mask
            mean_noise_span_length: a number

        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        num_nonnoise_tokens = length - num_noise_tokens
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        # num_noise_tokens should be less than num_noise_tokens and num_nonnoise_tokens
        num_noise_spans = int(np.round(
            min(num_noise_tokens, num_nonnoise_tokens) / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            self.rng.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(
            num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(
            num_nonnoise_tokens, num_noise_spans)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths],
                     axis=1), [num_noise_spans * 2]
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]

    def shift_tokens_right(self, input_ids: np.ndarray, pad_token_id: int, decoder_start_token_id: int) -> np.ndarray:
        """
        Shift input ids one token to the right.
        """
        shifted_input_ids = np.zeros_like(input_ids)
        # Shift input ids to the right
        shifted_input_ids[:, 1:] = input_ids[:, :-1]
        # Set the first token to be the decoder start token id
        shifted_input_ids[:, 0] = decoder_start_token_id

        # Replace any -100 in the array with the pad token id
        shifted_input_ids = np.where(
            shifted_input_ids == -100, pad_token_id, shifted_input_ids)
        return shifted_input_ids
