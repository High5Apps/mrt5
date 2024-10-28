# diagnostic_task_eval.py
# Author: Julie Kallini

import sys
sys.path.append('..')

import torch
import argparse
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import (
    get_task_dataset,
    load_model_from_path,
)

def calculate_seq_accuracy(labels, outputs):
    logits = outputs.logits
    # Get the predicted IDs
    predicted_ids = torch.argmax(logits, dim=-1)

    # Compare predicted_ids with the true labels
    correct_predictions = (predicted_ids == labels).all(dim=-1).sum().item()
    total_predictions = labels.shape[0]

    # Calculate accuracy
    accuracy = correct_predictions / total_predictions

    return accuracy


def calculate_token_accuracy(labels, outputs):
    logits = outputs.logits
    # Get the predicted IDs
    predicted_ids = torch.argmax(logits, dim=-1)

    # Compare predicted_ids with the true labels
    correct_predictions = (predicted_ids == labels).sum().item()

    # Calculate accuracy
    total_predictions = labels.numel()
    accuracy = correct_predictions / total_predictions

    return accuracy


def get_input_ids_and_labels(batch):
    input_ids = batch['input_ids'].squeeze(axis=1).to(device)
    labels = batch['labels'].squeeze(axis=1).to(device)
    return input_ids, labels


def byt5_compute_loss(model, batch):
    # Get input ids and labels
    input_ids, labels = get_input_ids_and_labels(batch)

    # Get model outputs
    outputs = model(input_ids=input_ids, labels=labels,
                    output_hidden_states=True)

    # Get accuracy scores
    token_accuracy = calculate_token_accuracy(labels, outputs)
    seq_accuracy = calculate_seq_accuracy(labels, outputs)

    # Return cross entropy loss, percent deleted tokens, and new sequence length
    return outputs.loss.item(), 0.0, \
        outputs.encoder_last_hidden_state.shape[1], \
        token_accuracy, seq_accuracy


def mrt5_compute_loss(model, batch):
    # Get input ids and labels
    input_ids, labels = get_input_ids_and_labels(batch)

    # Get model outputs
    outputs = model(
        input_ids=input_ids,
        labels=labels,
        hard_delete=True,
        output_hidden_states=True,
        deletion_threshold=args.deletion_threshold)

    # Get delete gate output
    delete_gate_output = outputs.delete_gate_output.squeeze(-1)

    # Compute percent deleted tokens
    batch_size, seq_len = input_ids.shape[0:2]
    num_deleted_tokens = (delete_gate_output < args.deletion_threshold / 2).sum()
    percent_deleted_tokens = num_deleted_tokens / (batch_size * seq_len) * 100

    # Get accuracy scores
    token_accuracy = calculate_token_accuracy(labels, outputs)
    seq_accuracy = calculate_seq_accuracy(labels, outputs)

    # Return cross entropy loss, percent deleted tokens, and new sequence length
    return outputs.loss.item(), percent_deleted_tokens.mean().item(), \
        outputs.encoder_last_hidden_state.shape[1], \
        token_accuracy, seq_accuracy


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
                        default=20000, help='Model checkpoint to load for evaluation.')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument('--deletion_threshold', type=float,
                        default=-15.0, help='Deletion gate threshold.')

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
        compute_loss_function = byt5_compute_loss
    elif args.model_type in ('MrT5', 'RandomT5', 'FixedT5'):
        compute_loss_function = mrt5_compute_loss
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
            # Compute the loss
            loss, percent_deleted_tokens, new_seq_len, token_accuracy, seq_accuracy = \
                compute_loss_function(model, batch)

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
