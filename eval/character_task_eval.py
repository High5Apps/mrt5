# character_task_eval.py
# Author: Julie Kallini

import sys
sys.path.append('..')

import time
import os
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import (
    get_task_dataset,
    load_model_from_path,
    CHAR_IIT_TASKS_AND_INFO,
    byt5_compute_metrics,
    mrt5_compute_metrics,
    bpt5_compute_metrics,
    MODEL_ARCHITECTURES,
)
from functools import partial

def get_input_ids_and_labels(batch):
    input_ids = batch['input_ids'].squeeze(axis=1).to(device)
    labels = batch['labels'].squeeze(axis=1).to(device)
    return input_ids, labels

def load_eval_dataset(task, split, batch_size):
    # Load the dataset
    dataset = get_task_dataset(task, split, iterable_dataset=False)
    dataset = dataset.with_format(type="torch")

    # Create DataLoader for the evaluation dataset
    dataloader = DataLoader(
        dataset, batch_size=batch_size)

    return dataloader


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Test eval metrics for ByT5/MrT5 models on character-level tasks.')
    parser.add_argument('model_name', type=str,
                        help='Name of model/run to load.')
    parser.add_argument('model_type',
                        type=str,
                        const='all',
                        nargs='?',
                        choices=MODEL_ARCHITECTURES,
                        help='Type of model architecture to evaluate.')
    parser.add_argument("task", type=str, choices=CHAR_IIT_TASKS_AND_INFO.keys(),
                        help="Task to evaluate the model on.")
    parser.add_argument('--checkpoint', type=int,
                        default=3000, help='Model checkpoint to load for evaluation.')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument('--deletion_threshold', type=float,
                        default=-15.0, help='Deletion gate threshold.')

    # Eval arguments
    parser.add_argument('--per_device_eval_batch_size', type=int,
                        default=64, help='Batch size per device during evaluation.')
    parser.add_argument('--hard_delete', action='store_true', help='Use hard deletion instead of soft deletion.')


    args = parser.parse_args()

    print("Loading model...")
    model = load_model_from_path(args.model_type, model_name=args.model_name,
                                 training_task=args.task, seed=args.random_seed, ckpt=args.checkpoint)

    # Move the model to the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Evaluation loop
    model.eval()

    # Determine loss function based on model type
    if args.model_type == 'T5':
        metrics_function = byt5_compute_metrics
    elif args.model_type == 'MrT5':
        metrics_function = partial(mrt5_compute_metrics,
                                   deletion_threshold=args.deletion_threshold,
                                   hard_delete=args.hard_delete)
    elif args.model_type == 'BPT5':
        metrics_function = bpt5_compute_metrics
    else:
        raise ValueError(
            "Model type must be 'T5' or 'MrT5'.")

    seq_accuracy_data = []
    percent_deleted_tokens_data = []
    runtime_data = []
    size_data = []

    SPLITS = [split for split in CHAR_IIT_TASKS_AND_INFO[args.task]['splits'] if 'test' in split]

    with torch.no_grad():
        for split in SPLITS:
            print(f"Evaluating on {args.task}, {split} split...")
            # Load the evaluation dataset
            eval_dataloader = load_eval_dataset(
                args.task, split, batch_size=args.per_device_eval_batch_size)

            # Initialize the total loss
            total_accuracy = 0.0
            total_percent_deleted_tokens = 0.0

            # Start the timer
            start_time = time.time()

            print(f"Number of batches: {len(eval_dataloader)}")
            print(f"Number of examples: {len(eval_dataloader.dataset)}")

            num_batches = len(eval_dataloader)
            for batch in tqdm(eval_dataloader):
                # Compute metrics
                input_ids, labels = get_input_ids_and_labels(batch)

                # Get metrics from the model
                _, percent_deleted_tokens, _, seq_accuracy, _ = \
                                metrics_function(model, input_ids, labels)

                # Update the total metrics
                total_accuracy += seq_accuracy
                total_percent_deleted_tokens += percent_deleted_tokens

            # End the timer
            end_time = time.time()
            eval_runtime = (end_time - start_time) / len(eval_dataloader.dataset) * 1000

            average_seq_accuracy = total_accuracy / num_batches * 100
            average_percent_deleted_tokens = total_percent_deleted_tokens / num_batches

            seq_accuracy_data.append(average_seq_accuracy)
            percent_deleted_tokens_data.append(
                average_percent_deleted_tokens)
            runtime_data.append(eval_runtime)
            size_data.append(len(eval_dataloader.dataset))

            print(f"Seq Accuracy: {average_seq_accuracy}")
            print(f"Percent deleted tokens: {average_percent_deleted_tokens}")
            print(f"Eval runtime: {eval_runtime} seconds")
            print()

        # Save the evaluation metrics to a CSV file
        eval_metrics = pd.DataFrame({
            'Split': SPLITS,
            'Eval Sequence Accuracy': seq_accuracy_data,
            'Eval Percent Deleted Tokens': percent_deleted_tokens_data,
            'Eval Runtime': runtime_data,
            'Size': size_data,
        })

        # Make directory for eval results
        dir_path = f"eval_results/character_tasks/{args.task}/{args.model_type}"
        os.makedirs(dir_path, exist_ok=True)

        outfile = dir_path + f"/{args.model_name}_seed{args.random_seed}.csv"
        eval_metrics.to_csv(outfile, index=False)

        print(f"Saved evaluation metrics to: {outfile}")
