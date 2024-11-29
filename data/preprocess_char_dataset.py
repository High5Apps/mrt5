# preprocess_char_dataset.py
# Author: Julie Kallini

import sys
sys.path.append('..')

import argparse
import json
import csv
import os
from utils import FINETUNE_DATASET_PATH, CHAR_IIT_TASKS, CHAR_IIT_TASKS_AND_INFO
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import Dataset


class CharacterTaskDataCollator:
    def __init__(self, tokenizer, input_seq_length=24, output_seq_length=16):
        self.tokenizer = tokenizer
        self.truncation = True
        self.padding = 'max_length'
        self.input_seq_length = input_seq_length
        self.output_seq_length = output_seq_length
        # 0: entailment, 1: contradiction, 2: neutral
        self.label_map = {0: '0', 1: '1', 2: '2'}

    def __call__(self, features=None):
        # Tokenize inputs and labels
        inputs_batch = self.tokenizer(features[0]['input'], padding="max_length", truncation=self.truncation,
                                      max_length=self.input_seq_length, return_tensors='pt')
        labels_batch = self.tokenizer(features[0]['output'], padding=self.padding, truncation=self.truncation,
                                      max_length=self.output_seq_length, return_tensors='pt')

        batch = {}
        batch['input_ids'] = inputs_batch["input_ids"]
        batch['attention_mask'] = inputs_batch["attention_mask"]
        batch['labels'] = labels_batch["input_ids"]

        return batch


def save_processed_dataset(dataset, filename):
    with open(filename, 'w') as f:
        for item in dataset:
            json.dump(item, f)
            f.write('\n')


def preprocess_dataset(dataset, collator):
    for example in tqdm(dataset, total=len(dataset)):
        processed_example = collator([example])
        yield {
            'input_ids': processed_example['input_ids'].tolist(),
            'attention_mask': processed_example['attention_mask'].tolist(),
            'labels': processed_example['labels'].tolist(),
        }


def stream_and_preprocess(dataset, collator, task, outfile):
    # Process datasets to files
    preprocessed_dataset = preprocess_dataset(dataset, collator)

    # Create dir if it doesn't exist
    dataset_path = f"{FINETUNE_DATASET_PATH}/char_iit_json/{task}/"
    if not os.path.exists(dataset_path):
        # Create a new directory because it does not exist
        os.makedirs(dataset_path)

    save_processed_dataset(preprocessed_dataset, dataset_path + outfile)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Preprocess character-level dataset for fine-tuning.')

    # Toy task arguments
    parser.add_argument(
        'task',
        choices=CHAR_IIT_TASKS,
        help='Character-level dataset to preprocess.'
    )
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

    def select_columns(filepath):
        # Initialize lists for the first two columns
        col_1 = []
        col_2 = []

        # Open the TSV file and read the data
        with open(filepath, newline='', encoding='utf-8') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')

            # Iterate over each row in the TSV file
            for row in reader:
                # Append the first two columns to the respective lists
                col_1.append(row[0])
                col_2.append(row[1])

        # Create a dictionary where each key is a column name and the value is a list of column data
        data_dict = {
            'input': col_1,
            'output': col_2
        }

        return data_dict

    CHAR_IIT_PATH = FINETUNE_DATASET_PATH + "/char_iit_data/"
    if not os.path.exists(CHAR_IIT_PATH):
        raise FileNotFoundError(
            f"Directory {CHAR_IIT_PATH} does not exist. Please download "
            + "the data from the char-iit github repository and move it "
            + "to this path first.")

    # Get task-specific arguments
    splits = CHAR_IIT_TASKS_AND_INFO[args.task]['splits']
    input_seq_length = CHAR_IIT_TASKS_AND_INFO[args.task]['input_seq_length']
    output_seq_length = CHAR_IIT_TASKS_AND_INFO[args.task]['output_seq_length']

    collator = CharacterTaskDataCollator(
        tokenizer, input_seq_length=input_seq_length, output_seq_length=output_seq_length)

    print("Task:", args.task)
    print("Splits:", splits)
    print("Input sequence length:", input_seq_length)
    print("Output sequence length:", output_seq_length)

    for split in splits:
        print(f"Processing {args.task} {split} split...")
        data = select_columns(CHAR_IIT_PATH + f"{args.task}_{split}.tsv")
        dataset = Dataset.from_dict(data)
        stream_and_preprocess(
            dataset, collator, args.task, f"{args.task}_{split if split != 'val' else 'validation'}.json")
