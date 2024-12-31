# diagnostic_task_eval.py
# Author: Julie Kallini

import sys
sys.path.append('..')

import torch
import argparse
from functools import partial
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import (
    get_task_dataset,
    load_model_from_path,
    byt5_compute_metrics,
    mrt5_compute_metrics,
)

def get_input_ids_and_labels(batch):
    input_ids = batch['input_ids'].squeeze(axis=1).to(device)
    labels = batch['labels'].squeeze(axis=1).to(device)
    return input_ids, labels


def load_eval_dataset(batch_size):
    # Load the dataset
    print(f"Loading {args.diagnostic_task} test dataset...")
    dataset = get_task_dataset(args.diagnostic_task, split="test", iterable_dataset=True)
    dataset = dataset.with_format(type="torch")

    # Create DataLoader for the evaluation dataset
    dataloader = DataLoader(
        dataset, batch_size=batch_size)

    return dataloader


if __name__ == "__main__":

    MODEL_CHOICES = ['T5', 'MrT5', 'DecoderBaselineT5', 'RandomT5', 'FixedT5']

    parser = argparse.ArgumentParser(
        description='Test basic evaluation metrics for ByT5/MrT5 models.')
    parser.add_argument('diagnostic_task', type=str)
    parser.add_argument('model_name', type=str,
                        help='Name of model/run to load.')
    parser.add_argument('model_type',
                        type=str,
                        const='all',
                        nargs='?',
                        choices=MODEL_CHOICES,
                        help='Type of model architecture to evaluate.')
    parser.add_argument('--num_batches', type=int,
                        default=250, help='Number of batches to evaluate.')
    parser.add_argument('--checkpoint', type=int,
                        default=30000, help='Model checkpoint to load for evaluation.')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument('--deletion_threshold', type=float,
                        default=-15.0, help='Deletion gate threshold.')
    parser.add_argument('--hard_delete', action='store_true', help='Use hard deletion instead of soft deletion.')

    # Eval arguments
    parser.add_argument('--per_device_eval_batch_size', type=int,
                        default=128, help='Batch size per device during evaluation.')

    args = parser.parse_args()

    print("Loading model...")
    model = load_model_from_path(args.model_type, model_name=args.model_name,
                                 training_task=args.diagnostic_task, seed=args.random_seed, ckpt=args.checkpoint)

    # Move the model to the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Evaluation loop
    model.eval()

    # Determine loss function based on model type
    if args.model_type == 'T5':
        metrics_function = byt5_compute_metrics
    elif args.model_type in ('MrT5', 'RandomT5', 'FixedT5'):
        metrics_function = partial(mrt5_compute_metrics,
                                   deletion_threshold=args.deletion_threshold,
                                   hard_delete=args.hard_delete)
    else:
        raise ValueError(
            f"Model type must be one of {', '.join(MODEL_CHOICES)}.")

    print()

    loss_data = []
    percent_deleted_tokens_data = []
    new_seq_len_data = []
    token_accuracy_data = []
    seq_accuracy_data = []

    with torch.no_grad():
        # Load the evaluation dataset
        eval_dataloader = load_eval_dataset(batch_size=args.per_device_eval_batch_size)

        # Initialize the total loss
        total_loss = 0.0
        total_percent_deleted_tokens = 0.0
        total_new_seq_len = 0.0
        total_token_accuracy = 0.0
        total_seq_accuracy = 0.0
        num_batches = 0

        for batch in tqdm(eval_dataloader, total=args.num_batches):

            input_ids, labels = get_input_ids_and_labels(batch)
            
            # Compute the loss
            loss, percent_deleted_tokens, new_seq_len, seq_accuracy, token_accuracy = \
                metrics_function(model, input_ids, labels)

            # Update the total metrics
            total_loss += loss
            total_percent_deleted_tokens += percent_deleted_tokens
            total_new_seq_len += new_seq_len
            total_token_accuracy += token_accuracy
            total_seq_accuracy += seq_accuracy

            # Update the running count of batches
            num_batches += 1
            if num_batches >= args.num_batches:
                break

        average_loss = total_loss / num_batches
        average_percent_deleted_tokens = total_percent_deleted_tokens / num_batches
        average_new_seq_len = total_new_seq_len / num_batches
        average_token_accuracy = total_token_accuracy / num_batches
        average_seq_accuracy = total_seq_accuracy / num_batches

        loss_data.append(average_loss)
        percent_deleted_tokens_data.append(
            average_percent_deleted_tokens)
        new_seq_len_data.append(average_new_seq_len)

        print(f"Eval cross entropy loss: {average_loss}")
        print(
            f"Eval percent deleted tokens: {average_percent_deleted_tokens}")
        print(f"Eval new sequence length: {average_new_seq_len}")
        print(f"Eval token accuracy: {average_token_accuracy}")
        print(f"Eval sequence accuracy: {average_seq_accuracy}")
        print(
            f"Examples evaluated: {args.per_device_eval_batch_size * num_batches}")
        print()
