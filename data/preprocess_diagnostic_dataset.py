# preprocess_diagnostic_dataset.py
# Author: Julie Kallini

import sys
sys.path.append('..')

from transformers import AutoTokenizer
from tqdm import tqdm
from utils import DIAGNOSTIC_TASKS, DIAGNOSTIC_DATASET_PATH
import json
import argparse
import string
import os
import re
import numpy as np  # Ensure to import numpy for Gaussian distribution


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

    def __call__(self, features=None):
        raise NotImplementedError(
            "This method should be implemented by subclasses")


class MergeABCDataCollator(DiagnosticDataCollator):
    def __init__(self, tokenizer, padding='max_length', max_length=128, seed=42, avg_abc_count=10):
        super().__init__(tokenizer, padding, max_length, seed)
        self.avg_abc_count = avg_abc_count  # Average number of times "ABC" will be placed
        # Initialize a random generator with a seed
        self.rng = np.random.default_rng(seed)

    def get_abc_indices(self, num_insertions, seq_len):
        result = []
        attempts = 0
        while len(result) < num_insertions:

            # Choose a random position to insert "ABC"
            # Start at 1 to exclude the "#" character
            candidate = self.rng.choice(range(seq_len - 3))

            # Check if this candidate is at least 3 away from all existing numbers in the result
            if all(abs(candidate - num) >= 3 for num in result):
                result.append(candidate)

            # If we have tried too many times, break the loop
            attempts += 1
            if attempts > 500:
                break

        return result

    def __call__(self, features=None):
        if features is None:
            features = [
                {'text': self.generate_random_string(self.max_length - 2)}]

        inputs_with_abc = []
        labels_with_d = []

        for text in [feature['text'] for feature in features]:

            # Determine how many times to insert "ABC" based on a Gaussian distribution
            num_insertions = max(
                1, int(self.rng.normal(loc=self.avg_abc_count, scale=self.avg_abc_count / 2)))

            # Get the indices where "ABC" will be inserted
            insertion_indices = self.get_abc_indices(num_insertions, len(text))

            # Insert "ABC" into the input text and "D" into the label text
            input_text = text
            for insert_pos in insertion_indices:
                input_text = (input_text[:insert_pos] + "ABC" +
                              input_text[insert_pos + 3:])

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

    def __init__(self, tokenizer, padding='max_length', max_length=128, seed=42, avg_cv_count=50):
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

    def get_cv_indices(self, num_insertions, seq_len):
        result = []
        attempts = 0
        while len(result) < num_insertions:
            # Choose a random position to insert the consonant-vowel pair
            candidate = self.rng.choice(range(seq_len - 2))

            # Check if this candidate is at least 2 away from all existing numbers in the result
            if all(abs(candidate - num) >= 2 for num in result):
                result.append(candidate)

            # If we have tried too many times, break the loop
            attempts += 1
            if attempts > 500:
                break

        return result

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
            insertion_indices = self.get_cv_indices(num_insertions, len(text))

            input_text = text
            for insert_pos in insertion_indices:
                # Generate a random consonant-vowel pair
                cv_pair = self.generate_random_cv()

                # Insert the consonant-vowel pair into the input text
                input_text = (input_text[:insert_pos] +
                              cv_pair + input_text[insert_pos + 2:])

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
    def __init__(self, tokenizer, padding='max_length', max_length=128, seed=42):
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
    def __init__(self, tokenizer, padding='max_length', max_length=128, seed=42):
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
                        default=128, help='Input sequence length for the model.')
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
