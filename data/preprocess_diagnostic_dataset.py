# preprocess_diagnostic_dataset.py
# Author: Julie Kallini

import sys
sys.path.append('..')

import numpy as np  # Ensure to import numpy for Gaussian distribution
import re
import os
import string
import argparse
import json
from utils import DIAGNOSTIC_TASKS, DIAGNOSTIC_DATASET_PATH
from tqdm import tqdm
from transformers import AutoTokenizer


class DiagnosticDataCollator:
    def __init__(self, tokenizer, padding='max_length', max_length=128, seed=42):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.rng = np.random.default_rng(seed)

    def generate_random_string(self, length):
        alpha_chars = string.ascii_lowercase + string.ascii_uppercase
        return ''.join(self.rng.choice(list(alpha_chars), size=length))

    def tokenize_inputs(self, features):
        inputs = ["#" + feature['text'] for feature in features]
        return self.tokenizer(inputs, padding=self.padding, truncation=True,
                              max_length=self.max_length, return_tensors='pt')

    def get_insertion_indices(self, num_insertions, seq_len, subseq_len):
        result = []
        attempts = 0
        while len(result) < num_insertions:
            # Choose a random position to insert the sequence
            candidate = self.rng.choice(range(seq_len - subseq_len))

            # Check if this candidate is at least subseq_len away from
            # all existing numbers in the result
            if all(abs(candidate - num) >= subseq_len for num in result):
                result.append(candidate)

            attempts += 1

            # If we have tried too many times, break the loop
            if attempts > 500:
                break

        return result

    def insert_subsequence(self, text, subseq, insert_pos):
        return text[:insert_pos] + subseq + text[insert_pos + len(subseq):]

    def __call__(self, features=None):
        raise NotImplementedError(
            "This method should be implemented by subclasses")


class MergeABCDataCollator(DiagnosticDataCollator):
    def __init__(self, tokenizer, padding='max_length', max_length=64, seed=42, avg_abc_count=5):
        super().__init__(tokenizer, padding, max_length, seed)
        self.avg_abc_count = avg_abc_count  # Average number of times "ABC" will be placed
        # Initialize a random generator with a seed
        self.rng = np.random.default_rng(seed)

    def insert_subsequences(self, text, subseq):
        # Determine how many times to insert subsequence
        num_insertions = max(
            1, int(self.rng.normal(loc=self.avg_abc_count, scale=self.avg_abc_count / 2)))

        # Get the indices where the subsequence will be inserted
        insertion_indices = self.get_insertion_indices(
            num_insertions, len(text), len(subseq))

        # Insert the subsequences into the input text
        input_text = text
        for insert_pos in insertion_indices:
            input_text = self.insert_subsequence(
                input_text, subseq, insert_pos)

        return input_text

    def __call__(self, features=None):
        if features is None:
            features = [
                {'text': self.generate_random_string(self.max_length - 2)}]

        inputs_with_abc = []
        labels_with_d = []

        for text in [feature['text'] for feature in features]:

            # Insert ABC sequences into text
            input_text = self.insert_subsequences(text, "ABC")

            # Replace ABC with D in the labels
            label_text = re.sub(r'ABC', 'D', input_text)

            inputs_with_abc.append("#" + input_text)
            labels_with_d.append("#" + label_text)

        # Tokenize inputs and labels
        inputs_batch = self.tokenizer(inputs_with_abc, padding=self.padding, truncation=True,
                                      max_length=self.max_length, return_tensors='pt')
        labels_batch = self.tokenizer(labels_with_d, padding=self.padding, truncation=True,
                                      max_length=self.max_length, return_tensors='pt')

        batch = {}
        batch['input_ids'] = inputs_batch["input_ids"]
        batch['labels'] = labels_batch["input_ids"]

        return batch


class ContextualVowelRemovalDataCollator(DiagnosticDataCollator):

    def __init__(self, tokenizer, padding='max_length', max_length=64, seed=42, avg_cv_count=25):
        super().__init__(tokenizer, padding, max_length, seed)
        self.VOWELS = set("aeiouAEIOU")
        self.LOWERCASE_CONSONANTS = set(
            string.ascii_lowercase).difference(self.VOWELS)
        self.CONSONANTS = set(string.ascii_lowercase +
                              string.ascii_uppercase).difference(self.VOWELS)

        # Average number of times the consonant-vowel (CV) pair will be placed
        self.avg_cv_count = avg_cv_count
        # Initialize a random generator with a seed
        self.rng = np.random.default_rng(seed)

    def generate_random_cv(self):
        # Generate a random consonant followed by a random vowel
        consonant = self.rng.choice(list(self.CONSONANTS))
        vowel = self.rng.choice(list(self.VOWELS))
        return consonant + vowel

    def __call__(self, features=None):
        if features is None:
            features = [
                {'text': self.generate_random_string(self.max_length - 2)}]

        inputs = []
        labels = []

        for text in [feature['text'] for feature in features]:
            # Determine how many times to insert a consonant-vowel pair based on a Gaussian distribution
            num_insertions = max(
                1, int(self.rng.normal(loc=self.avg_cv_count, scale=self.avg_cv_count / 2)))

            # Get the indices where the consonant-vowel pairs will be inserted
            insertion_indices = self.get_insertion_indices(
                num_insertions, len(text), 2)

            input_text = text
            for insert_pos in insertion_indices:
                # Generate a random consonant-vowel pair
                cv_pair = self.generate_random_cv()

                # Insert the consonant-vowel pair into the input text
                input_text = self.insert_subsequence(
                    input_text, cv_pair, insert_pos)

            # Replace each consonant-vowel pair with the consonant
            label_text = re.sub(
                f"([{''.join(list(self.LOWERCASE_CONSONANTS))}])" +
                f"[{''.join(list(self.VOWELS))}]", r'\1', input_text)

            inputs.append("#" + input_text)
            labels.append("#" + label_text)

        # Tokenize inputs and labels
        inputs_batch = self.tokenizer(inputs, padding=self.padding, truncation=True,
                                      max_length=self.max_length, return_tensors='pt')
        labels_batch = self.tokenizer(labels, padding=self.padding, truncation=True,
                                      max_length=self.max_length, return_tensors='pt')

        batch = {}
        batch['input_ids'] = inputs_batch["input_ids"]
        batch['labels'] = labels_batch["input_ids"]

        return batch


class VowelRemovalDataCollator(DiagnosticDataCollator):
    def __init__(self, tokenizer, padding='max_length', max_length=64, seed=42):
        super().__init__(tokenizer, padding, max_length, seed)

    def __call__(self, features=None):
        if features is None:
            features = [
                {'text': self.generate_random_string(self.max_length - 2)}]

        batch = self.tokenize_inputs(features)

        # Create labels by removing vowels directly
        label_texts = []
        for input_text in ["#" + feature['text'] for feature in features]:
            # Remove vowels from the text
            # Only include as many bytes as present in the input's max length
            no_vowel_text = re.sub(
                r'[aeiouAEIOU]', '', input_text[:self.max_length - 1])
            label_texts.append(no_vowel_text)

        # Tokenize target labels
        labels = self.tokenizer(text_target=label_texts, padding=self.padding,
                                truncation=True, max_length=self.max_length, return_tensors='pt')
        batch['labels'] = labels["input_ids"]

        return batch


class CopyDataCollator(DiagnosticDataCollator):
    def __init__(self, tokenizer, padding='max_length', max_length=64, seed=42):
        super().__init__(tokenizer, padding, max_length, seed)

    def __call__(self, features=None):
        if features is None:
            features = [{'text': self.generate_random_string(self.max_length)}]

        batch = self.tokenize_inputs(features)
        batch['labels'] = batch["input_ids"]
        return batch


def preprocess_and_save_dataset(n, collator, split='train'):

    def preprocess_dataset(n, collator):
        for _ in tqdm(range(n)):
            processed_example = collator()
            yield {
                'input_ids': processed_example['input_ids'].tolist(),
                'labels': processed_example['labels'].tolist(),
            }

    def save_dataset(dataset, filename):
        with open(filename, 'w') as f:
            for item in dataset:
                json.dump(item, f)
                f.write('\n')

    # Name of the output file
    uid = f"-{args.uid}" if args.uid != "" else ""
    outfile = f"{DIAGNOSTIC_DATASET_PATH}/diagnostic-{args.task}-{split}{uid}.json"

    # Process datasets to files
    preprocessed_dataset = preprocess_dataset(n, collator)

    # Save the processed dataset to a file
    save_dataset(preprocessed_dataset, outfile)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Generate a diagnostic dataset JSON.')

    # Diagnostic task arguments
    parser.add_argument(
        'task',
        choices=DIAGNOSTIC_TASKS,
        help='Diagnostic task to train the model on.'
    )

    # Data arguments
    parser.add_argument('--input_seq_length', type=int,
                        default=64, help='Input sequence length for the model.')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument("--train_n", type=int, default=1280000,
                        help="Number of training examples to sample")
    parser.add_argument("--eval_n", type=int, default=16000,
                        help="Number of evaluation examples to sample")
    parser.add_argument("--uid", type=str, default="",
                        help="Unique identifier for the output files")
    parser.add_argument("--split", type=str, default=None,
                        help="Split to preprocess (train, validation, test), if specified, only preprocess the split")
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

    if args.task == 'copy':
        collator = CopyDataCollator(
            tokenizer, max_length=args.input_seq_length, seed=args.random_seed)
    elif args.task == 'vowel_removal':
        collator = VowelRemovalDataCollator(
            tokenizer, max_length=args.input_seq_length, seed=args.random_seed)
    elif args.task == 'contextual_vowel_removal':
        collator = ContextualVowelRemovalDataCollator(
            tokenizer, max_length=args.input_seq_length, seed=args.random_seed)
    elif args.task == 'merge_ABC':
        collator = MergeABCDataCollator(
            tokenizer, max_length=args.input_seq_length, seed=args.random_seed)
    else:
        raise ValueError(
            f"Diagnostic task must be one of {', '.join(DIAGNOSTIC_TASKS)}.")

    # Create diagnostic dataset path if it does not exist
    os.makedirs(DIAGNOSTIC_DATASET_PATH, exist_ok=True)

    if args.split is not None:
        n_samples = args.train_n if args.split == "train" else args.eval_n
        print(f"Preprocessing {args.task} {args.split} set...")
        preprocess_and_save_dataset(n_samples, collator, args.split)
    else:
        print(f"Preprocessing {args.task} train set...")
        preprocess_and_save_dataset(args.train_n, collator, "train")
        print(f"Preprocessing {args.task} validation set...")
        preprocess_and_save_dataset(args.eval_n, collator, "validation")
        print(f"Preprocessing {args.task} test set...")
        preprocess_and_save_dataset(args.eval_n, collator, "test")
