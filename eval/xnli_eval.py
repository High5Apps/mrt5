# xnli_eval.py
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
from transformers import AutoTokenizer
from datasets import load_dataset
from utils import (
    SUBSET_LANGUAGES as LANGUAGES,
    load_model_from_path,
    mrt5_compute_metrics,
    byt5_compute_metrics,
    bpt5_compute_metrics,
    MODEL_ARCHITECTURES,
)
from data.data_collator_finetuning import XNLIDataCollator
from functools import partial

def load_eval_dataset(language, batch_size):
    # Load tokenizer for the model
    tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')

    # Initialize the XNLI data collator
    collator = XNLIDataCollator(tokenizer=tokenizer)

    # Load XNLI test set from Hugging Face
    dataset = load_dataset("xnli", language, split="test")

    # Create DataLoader with XNLIDataCollator
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)

    return dataloader


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Test ByT5/MrT5 models on XNLI task.')
    parser.add_argument('model_name', type=str,
                        help='Name of model/run to load.')
    parser.add_argument('model_type',
                        type=str,
                        const='all',
                        nargs='?',
                        choices=MODEL_ARCHITECTURES,
                        help='Type of model architecture to evaluate.')
    parser.add_argument('--checkpoint', type=int,
                        default=3000, help='Model checkpoint to load for evaluation.')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument('--deletion_threshold', type=float,
                        default=-15.0, help='Deletion gate threshold.')
    parser.add_argument('--hard_delete', action='store_true', help='Use hard deletion instead of soft deletion.')
    parser.add_argument('--per_device_eval_batch_size', type=int,
                        default=64, help='Batch size per device during evaluation.')


    args = parser.parse_args()

    print("Loading model...")
    model = load_model_from_path(args.model_type, model_name=args.model_name,
                                 training_task=f"xnli", seed=args.random_seed, ckpt=args.checkpoint)

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

    with torch.no_grad():
        for language in LANGUAGES:
            print(f"Evaluating on {LANGUAGES[language]}...")
            # Load the evaluation dataset
            eval_dataloader = load_eval_dataset(
                language, batch_size=args.per_device_eval_batch_size)

            # Initialize the total loss
            total_accuracy = 0.0
            total_percent_deleted_tokens = 0.0

            print(f"Number of batches: {len(eval_dataloader)}")
            print(f"Number of examples: {len(eval_dataloader.dataset)}")

            # Start the timer
            start_time = time.time()

            num_batches = len(eval_dataloader)
            for batch in tqdm(eval_dataloader):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                # Compute metrics
                acc, percent_deleted_tokens = metrics_function(model, input_ids, labels)

                # Update the total metrics
                total_accuracy += acc
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
            'Language': list(LANGUAGES.values()),
            'Language Code': list(LANGUAGES.keys()),
            'Eval Sequence Accuracy': seq_accuracy_data,
            'Eval Percent Deleted Tokens': percent_deleted_tokens_data,
            'Eval Runtime': runtime_data,
            'Size': size_data,
        })

        # Make directory for eval results
        os.makedirs(
            f"eval_results/xnli/{args.model_type}", exist_ok=True)

        outfile = f"eval_results/xnli/{args.model_type}/{args.model_name}_seed{args.random_seed}.csv"
        eval_metrics.to_csv(outfile, index=False)

        print(f"Saved evaluation metrics to: {outfile}")
