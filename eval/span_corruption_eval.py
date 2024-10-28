# span_corruption_eval.py
# Author: Julie Kallini

import sys
sys.path.append('..')

from utils import (
    SUBSET_LANGUAGES as LANGUAGES,
    get_task_dataset,
    load_model_from_path,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import argparse
import torch
import os
import time


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

    # Return cross entropy loss, percent deleted tokens, and new sequence length
    return outputs.loss.item(), 0.0, outputs.encoder_last_hidden_state.shape[1]


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
    num_deleted_tokens = (delete_gate_output <
                          args.deletion_threshold / 2).sum()
    percent_deleted_tokens = num_deleted_tokens / (batch_size * seq_len) * 100

    # Return cross entropy loss, percent deleted tokens, and new sequence length
    return outputs.loss.item(), percent_deleted_tokens.mean().item(), outputs.encoder_last_hidden_state.shape[1]


def decoder_baseline_compute_loss(model, batch):
    # Get input ids and labels
    input_ids, labels = get_input_ids_and_labels(batch)

    batch_size, _ = input_ids.shape

    # Create a dummy input_ids tensor to pass to the model
    input_ids = torch.tensor([[1]]*batch_size).to(input_ids.device)
    outputs = model(input_ids=input_ids, labels=labels,
                    output_hidden_states=True)

    # Return cross entropy loss, percent deleted tokens, and new sequence length
    return outputs.loss.item(), 100.0, outputs.encoder_last_hidden_state.shape[1]


def load_eval_dataset(language, batch_size):
    # Load the dataset
    print(f"----EVALUATION FOR LANGUAGE: {LANGUAGES[language]}----")
    print(f"Loading {LANGUAGES[language]} ({language}) test dataset...")
    dataset = get_task_dataset(
        "span_corruption", "test", language=language, iterable_dataset=True)
    dataset = dataset.with_format(type="torch")

    # Create DataLoader for the evaluation dataset
    dataloader = DataLoader(
        dataset, batch_size=batch_size)

    return dataloader


if __name__ == "__main__":

    MODEL_CHOICES = ['T5', 'MrT5', 'DecoderBaselineT5', 'RandomT5', 'FixedT5']

    parser = argparse.ArgumentParser(
        description='Test span corruption evaluation metrics for ByT5/MrT5 models.')
    parser.add_argument('model_name', type=str,
                        help='Name of model/run to load.')
    parser.add_argument('model_type',
                        type=str,
                        const='all',
                        nargs='?',
                        choices=MODEL_CHOICES,
                        help='Type of model architecture to evaluate.')
    parser.add_argument('--num_batches', type=int,
                        default=1000, help='Number of batches to evaluate.')
    parser.add_argument('--checkpoint', type=int,
                        default=3000, help='Model checkpoint to load for evaluation.')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument('--deletion_threshold', type=float,
                        default=-15.0, help='Deletion gate threshold.')

    # Eval arguments
    parser.add_argument('--per_device_eval_batch_size', type=int,
                        default=64, help='Batch size per device during evaluation.')

    # Eval arguments
    parser.add_argument('--per_sample_eval',
                        action='store_true', help='Evaluate English per-sample.')

    parser.add_argument('--multilingual_model',
                        action='store_true', help='Evaluate multilingual model.')

    parser.add_argument('--en_only', action='store_true',
                        help='Evaluate English only.')

    args = parser.parse_args()

    multilingual = "" if not args.multilingual_model else "_multilingual"

    print("Loading model...")
    model = load_model_from_path(args.model_type, model_name=args.model_name,
                                 training_task=f"span_corruption{multilingual}", seed=args.random_seed, ckpt=args.checkpoint)

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
    elif args.model_type == 'DecoderBaselineT5':
        compute_loss_function = decoder_baseline_compute_loss
    else:
        raise ValueError(
            f"Model type must be {', '.join(MODEL_CHOICES)}.")

    print()

    loss_data = []
    percent_deleted_tokens_data = []
    new_seq_len_data = []
    runtime_data = []

    with torch.no_grad():
        if args.per_sample_eval:
            # Load the evaluation dataset
            eval_dataloader = load_eval_dataset("en", batch_size=1)

            loss_data = []
            percent_deleted_tokens_data = []
            new_seq_len_data = []
            runtime_data = []
            input_ids_data = []
            labels_data = []

            i = 0
            for batch in tqdm(eval_dataloader, total=args.num_batches):
                # Start the timer
                start_time = time.time()

                # Compute the loss
                loss, percent_deleted_tokens, new_seq_len = compute_loss_function(
                    model, batch)

                # End the timer
                end_time = time.time()
                eval_runtime = end_time - start_time

                loss_data.append(loss)
                percent_deleted_tokens_data.append(percent_deleted_tokens)
                new_seq_len_data.append(new_seq_len)
                runtime_data.append(eval_runtime)
                input_ids_data.append(batch['input_ids'][0].tolist())
                labels_data.append(batch['labels'][0].tolist())

                i += 1
                if i >= args.num_batches:
                    break

            # Save the evaluation metrics to a CSV file
            eval_metrics = pd.DataFrame({
                'Eval Cross Entropy Loss': loss_data,
                'Eval Percent Deleted Tokens': percent_deleted_tokens_data,
                'Eval New Sequence Length': new_seq_len_data,
                'Eval Runtime': runtime_data,
                'Input IDs': input_ids_data,
                'Labels': labels_data
            })

            # Make directory for eval results
            os.makedirs(
                f"eval_results/span_corruption_per_sample/{args.model_type}", exist_ok=True)

            outfile = f"eval_results/span_corruption_per_sample/{args.model_type}/{args.model_name}_seed{args.random_seed}.csv"
            eval_metrics.to_csv(outfile, index=False)

            print(f"Total examples evaluated: {i}")
            print(f"Saved per-sample evaluation metrics to: {outfile}")

        else:
            if args.en_only:
                LANGUAGES = {'en': 'English'}
            for language in LANGUAGES:
                # Load the evaluation dataset
                eval_dataloader = load_eval_dataset(
                    language, batch_size=args.per_device_eval_batch_size)

                # Initialize the total loss
                total_loss = 0.0
                total_percent_deleted_tokens = 0.0
                total_new_seq_len = 0.0
                num_batches = 0

                # Start the timer
                start_time = time.time()

                for batch in tqdm(eval_dataloader, total=args.num_batches):
                    # Compute the loss
                    loss, percent_deleted_tokens, new_seq_len = compute_loss_function(
                        model, batch)

                    # Update the total metrics
                    total_loss += loss
                    total_percent_deleted_tokens += percent_deleted_tokens
                    total_new_seq_len += new_seq_len

                    # Update the running count of batches
                    num_batches += 1
                    if num_batches >= args.num_batches:
                        break

                # End the timer
                end_time = time.time()
                eval_runtime = end_time - start_time

                average_loss = total_loss / num_batches
                average_percent_deleted_tokens = total_percent_deleted_tokens / num_batches
                average_new_seq_len = total_new_seq_len / num_batches

                loss_data.append(average_loss)
                percent_deleted_tokens_data.append(
                    average_percent_deleted_tokens)
                new_seq_len_data.append(average_new_seq_len)
                runtime_data.append(eval_runtime)

                print(f"Eval cross entropy loss: {average_loss}")
                print(
                    f"Eval percent deleted tokens: {average_percent_deleted_tokens}")
                print(f"Eval new sequence length: {average_new_seq_len}")
                print(f"Eval runtime: {eval_runtime} seconds")
                print(
                    f"Examples evaluated: {args.per_device_eval_batch_size * num_batches}")
                print()

            # Save the evaluation metrics to a CSV file
            eval_metrics = pd.DataFrame({
                'Language': list(LANGUAGES.values()),
                'Language Code': list(LANGUAGES.keys()),
                'Eval Cross Entropy Loss': loss_data,
                'Eval Percent Deleted Tokens': percent_deleted_tokens_data,
                'Eval New Sequence Length': new_seq_len_data,
                'Eval Runtime': runtime_data,
            })

            # Make directory for eval results
            os.makedirs(
                f"eval_results/span_corruption{multilingual}/{args.model_type}", exist_ok=True)

            en_only = "_en_only" if args.en_only else ""
            outfile = f"eval_results/span_corruption{multilingual}/{args.model_type}/{args.model_name}_seed{args.random_seed}{en_only}.csv"
            eval_metrics.to_csv(outfile, index=False)

            print(f"Saved evaluation metrics to: {outfile}")
