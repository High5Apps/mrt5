# preprocess_lm_dataset.py
# Author: Julie Kallini

import sys
sys.path.append('..')

from datasets import load_dataset
from transformers import AutoTokenizer
from data_collator_for_t5_mlm import (
    DataCollatorForT5MLM,
    t5_mlm_tokenize_function,
    compute_input_and_target_lengths
)
from utils import SUBSET_LANGUAGES as LANGUAGES, LM_DATASET_PATH
from tqdm import tqdm
import numpy as np
import json
import argparse
import os


def save_processed_dataset(dataset, filename):
    with open(filename, 'w') as f:
        for item in dataset:
            json.dump(item, f)
            f.write('\n')


def preprocess_dataset(dataset, num_examples, collator, split, language):

    # Skip the first validation examples for English
    if split == "test" and language == "en":
        max_examples = num_examples + args.eval_n
    else:
        max_examples = num_examples
    
    # Iterate over the dataset and preprocess examples
    for i, example in tqdm(enumerate(dataset), total=max_examples):
        # Skip the first validation examples for English
        if split == "test" and language == "en" and i < args.eval_n:
            continue

        # Process the example with the data collator
        processed_example = collator([example])

        # For test set only, decode the input tokens to get the characters
        decoded_input_ids = None
        if split == "test":
            decoded_input_ids = tokenizer.convert_ids_to_tokens(processed_example['input_ids'].squeeze())
        
        yield {
            'input_ids': processed_example['input_ids'].tolist(),
            'labels': processed_example['labels'].tolist(),
            'decoder_input_ids': processed_example['decoder_input_ids'].tolist(),
            'decoded_input_ids': decoded_input_ids,
        }

        if i >= max_examples:
            break


def stream_and_preprocess(n, collator, language='en', uid="", split='train', seed=42):

    # Stream the C4 dataset with the specified language and split
    stream_split = "validation" if split == "test" else split
    streaming_dataset = load_dataset(
        'allenai/c4', language, split=stream_split, streaming=True)

    # Tokenize the streamed dataset
    tokenized_dataset = streaming_dataset.map(
        tokenize_function, batched=True, remove_columns=["text", "timestamp", "url"])

    # Process datasets to files
    preprocessed_dataset = preprocess_dataset(tokenized_dataset, n, collator, split, language)

    # Save the processed dataset to a file
    if uid != "":
        uid = f"-{uid}"
    outfile = f"{LM_DATASET_PATH}/mc4-{language}-{split}{uid}.json"
    save_processed_dataset(preprocessed_dataset, outfile)


def preprocess_dataset_multilingual(datasets, num_examples, collator, rng):
    
    with tqdm(total=num_examples) as pbar:
        i = 0
        while i < num_examples:

            # Sample an example from a language dataset
            example = None
            while example is None:
                dataset_idx = rng.integers(0, len(datasets))
                dataset = datasets[dataset_idx]
                example = next(dataset, None)

            # Process the example with the data collator
            processed_example = collator([example])

            yield {
                'input_ids': processed_example['input_ids'].tolist(),
                'labels': processed_example['labels'].tolist(),
                'decoder_input_ids': processed_example['decoder_input_ids'].tolist(),
                'language': list(LANGUAGES.keys())[dataset_idx],
            }
            pbar.update(1)
            i += 1

def stream_and_preprocess_multilingual(n, collator, uid="", seed=42):

    datasets = []
    for lang in LANGUAGES.keys():
        # Stream the C4 dataset with the specified language and split
        streaming_dataset = load_dataset(
            'allenai/c4', lang, split="train", streaming=True)

        # Tokenize the streamed dataset
        tokenized_dataset = streaming_dataset.map(
            tokenize_function, batched=True, remove_columns=["text", "timestamp", "url"])
        
        datasets.append(iter(tokenized_dataset))

    # Process datasets to files
    rng = np.random.default_rng(seed)
    preprocessed_dataset = preprocess_dataset_multilingual(datasets, n, collator, rng)

    # Save the processed dataset to a file
    if uid != "":
        uid = f"-{uid}"
    outfile = f"{LM_DATASET_PATH}/mc4-multilingual-train{uid}.json"
    save_processed_dataset(preprocessed_dataset, outfile)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Preprocess mC4 dataset for span corruption training/eval.')
    parser.add_argument('--noise_density', type=float,
                        default=0.15, help='Noise density for data collation.')
    parser.add_argument('--mean_noise_span_length', type=float,
                        default=20.0, help='Mean noise span length.')
    parser.add_argument('--input_seq_length', type=int,
                        default=1024, help='Input sequence length for the model.')
    parser.add_argument('--decoder_start_token_id', type=int,
                        default=0, help='Decoder start token ID.')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument("--train_n", type=int, default=10240000,
                        help="Number of training examples to sample")
    parser.add_argument("--eval_n", type=int, default=16000,
                        help="Number of evaluation examples to sample")
    parser.add_argument("--test_n", type=int, default=160000,
                        help="Number of evaluation examples to sample")
    parser.add_argument("--uid", type=str, default="",
                        help="Unique identifier for the output files")
    parser.add_argument("--debug_mode", action='store_true')
    parser.add_argument("--en_only", action='store_true')
    parser.add_argument("--split", type=str, default=None,
                        help="Split to preprocess (train, validation, test), if specified, only preprocess the split")
    parser.add_argument("--multilingual", action='store_true')
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

    # Compute input and target lengths
    expanded_inputs_length, targets_length = compute_input_and_target_lengths(
        inputs_length=args.input_seq_length,
        noise_density=args.noise_density,
        mean_noise_span_length=args.mean_noise_span_length,
    )

    # Define the tokenization function
    def tokenize_function(examples):
        return t5_mlm_tokenize_function(examples, expanded_inputs_length, tokenizer)

    # Initialize the data collator
    collator = DataCollatorForT5MLM(
        tokenizer=tokenizer,
        noise_density=args.noise_density,
        mean_noise_span_length=args.mean_noise_span_length,
        input_length=args.input_seq_length,
        target_length=targets_length,
        pad_token_id=tokenizer.pad_token_id,
        decoder_start_token_id=args.decoder_start_token_id,
        rng=np.random.default_rng(args.random_seed),
        debug_mode=args.debug_mode,
    )

    # Create diagnostic dataset path if it does not exist
    os.makedirs(LM_DATASET_PATH, exist_ok=True)

    if args.multilingual:
        print(f"Preprocessing multilingual train set...")
        stream_and_preprocess_multilingual(args.train_n, collator, args.uid)

    else:
        if args.en_only:
            LANGUAGES = {"en": "English"}

        # Iterate over languages and sample training and validation sets
        for i, l in enumerate(LANGUAGES.keys()):

            if args.split is not None:
                n_samples = args.train_n if args.split == "train" else args.eval_n
                print(f"Preprocessing {LANGUAGES[l]} ({l}) {args.split} set...")
                stream_and_preprocess(n_samples, collator, l, args.uid, args.split)
            else:
                print(f"{i+1}: {LANGUAGES[l]} ({l})")
                print(f"Preprocessing {LANGUAGES[l]} ({l}) train set...")
                stream_and_preprocess(args.train_n, collator, l, args.uid, "train")
                print(f"Sampling {LANGUAGES[l]} ({l}) validation set...")
                stream_and_preprocess(args.eval_n, collator, l, args.uid, "validation")
                print(f"Sampling {LANGUAGES[l]} ({l}) test set...")
                stream_and_preprocess(args.test_n, collator, l, args.uid, "test")

