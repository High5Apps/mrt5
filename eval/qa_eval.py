# qa_eval.py
# Author: Julie Kallini
#
# The methods normalize_answer, f1_score, exact_match_score,
# metric_max_over_ground_truths, and evaluate_qa are adapted
# from the official evaluation script of SQuAD v1.1:
# https://raw.githubusercontent.com/allenai/bi-att-flow/master/squad/evaluate-v1.1.py

import sys
sys.path.append('..')

from functools import partial
from data.data_collator_finetuning import QADataCollator
from utils import (
    XQUAD_LANGUAGES,
    TYDIQA_LANGUAGES,
    load_model_from_path,
    mrt5_compute_metrics,
    byt5_compute_metrics,
    bpt5_compute_metrics,
    MODEL_ARCHITECTURES,
)
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from collections import Counter
from tqdm import tqdm
import pandas as pd
import re
import string
import argparse
import torch
import os
import time


def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate_qa(prediction, ground_truths):
    exact_match = metric_max_over_ground_truths(
        exact_match_score, prediction, ground_truths)
    f1 = metric_max_over_ground_truths(
        f1_score, prediction, ground_truths)
    return exact_match, f1


def byt5_compute_metrics(model, input_ids, ground_truths):

    # Get model outputs
    outputs = model.generate(
        input_ids=input_ids,
        max_length=1024,
        output_hidden_states=True,
        return_dict_in_generate=True)

    # Decode the prediction and get the evaluation metrics
    prediction = tokenizer.decode(
        outputs['sequences'][0], skip_special_tokens=True)
    exact_match, f1 = evaluate_qa(prediction, ground_truths[0])

    # Return cross entropy loss, accuracy, and percent deleted tokens
    return 0.0, exact_match, f1


def mrt5_compute_metrics(model, input_ids, ground_truths, deletion_threshold, hard_delete=True):

    # Get model outputs
    outputs = model.generate(
        input_ids=input_ids,
        max_length=1024,
        output_hidden_states=True,
        return_dict_in_generate=True,
        hard_delete=hard_delete,
        deletion_threshold=deletion_threshold)

    # Decode the prediction and get the evaluation metrics
    prediction = tokenizer.decode(
        outputs['sequences'][0], skip_special_tokens=True)
    exact_match, f1 = evaluate_qa(prediction, ground_truths)

    # Get the new sequence length
    percent_deleted_tokens = (1.0 - outputs['encoder_hidden_states'][-1].shape[1] / \
        input_ids.shape[1]) * 100

    # Return cross entropy loss, accuracy, and percent deleted tokens
    return percent_deleted_tokens.item(), exact_match, f1


def bp_canine_compute_metrics(model, input_ids, ground_truths):

    # Get model outputs
    outputs = model.generate(
        input_ids=input_ids,
        max_length=1024,
        output_hidden_states=True,
        return_dict_in_generate=True)

    # Decode the prediction and get the evaluation metrics
    prediction = tokenizer.decode(
        outputs['sequences'][0], skip_special_tokens=True)
    exact_match, f1 = evaluate_qa(prediction, ground_truths)

    # Get the new sequence length
    percent_deleted_tokens = (1.0 - outputs['encoder_hidden_states'][-1].shape[1] / \
        input_ids.shape[1]) * 100

    # Return cross entropy loss, accuracy, and percent deleted tokens
    return percent_deleted_tokens.item(), exact_match, f1


def bpt5_compute_metrics(model, input_ids, ground_truths):
    return bp_canine_compute_metrics(model, input_ids, ground_truths)


def canine_compute_metrics(model, input_ids, ground_truths):
    return bp_canine_compute_metrics(model, input_ids, ground_truths)


def load_eval_dataset(language, batch_size):
    # Initialize the QA data collator
    collator = QADataCollator(tokenizer=tokenizer)

    if args.task == 'xquad':
        # Load XQUAD test set from Hugging Face
        dataset = load_dataset(
            "google/xquad", f"xquad.{language}", split="validation")
    elif args.task == 'tydiqa':
        # Load TYDIQA test set from Hugging Face
        dataset = load_dataset('tydiqa', 'secondary_task', split='validation')
        # Filter by language
        dataset = dataset.filter(lambda example: example['id'].startswith(
            TYDIQA_LANGUAGES[language].lower()))

    # Create DataLoader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collator)

    return dataloader


def eval_loop(eval_dataloader):
    # Initialize the total loss
    total_f1 = 0.0
    total_exact_match = 0.0
    total_percent_deleted_tokens = 0.0

    print(f"Number of batches: {len(eval_dataloader)}")
    print(f"Number of examples: {len(eval_dataloader.dataset)}")

    # Start the timer
    start_time = time.time()

    num_batches = len(eval_dataloader)
    for batch in tqdm(eval_dataloader):
        input_ids = batch["input_ids"].to(device)
        all_answers = batch["all_answers"]

        # Compute metrics
        percent_deleted_tokens, exact_match, f1 = metrics_function(
            model, input_ids, all_answers)

        # Update the total metrics
        total_exact_match += exact_match
        total_f1 += f1
        total_percent_deleted_tokens += percent_deleted_tokens

    # End the timer
    end_time = time.time()
    eval_runtime = (end_time - start_time) / \
        len(eval_dataloader.dataset) * 1000

    average_exact_match = total_exact_match / num_batches * 100
    average_f1 = total_f1 / num_batches * 100
    average_percent_deleted_tokens = total_percent_deleted_tokens / num_batches

    print(f"Exact match: {average_exact_match}")
    print(f"F1: {average_f1}")
    print(f"Percent deleted tokens: {average_percent_deleted_tokens}")
    print(f"Eval runtime: {eval_runtime} seconds")
    print()

    return average_exact_match, average_f1, average_percent_deleted_tokens, eval_runtime


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Test ByT5/MrT5 models on question answering benchmarks.')
    parser.add_argument('task', type=str, choices=['xquad', 'tydiqa'],)
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

    # Eval arguments
    parser.add_argument('--hard_delete', action='store_true',
                        help='Use hard deletion instead of soft deletion.')

    args = parser.parse_args()

    print("Loading model...")
    model = load_model_from_path(args.model_type, model_name=args.model_name,
                                 training_task=args.task, seed=args.random_seed, ckpt=args.checkpoint)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')

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
    elif args.model_type == 'CanineT5':
        metrics_function = canine_compute_metrics
    else:
        raise ValueError(
            "Model type must be 'T5' or 'MrT5'.")

    exact_match_data = []
    f1_data = []
    percent_deleted_tokens_data = []
    runtime_data = []
    size_data = []

    with torch.no_grad():
        LANGUAGES = XQUAD_LANGUAGES if args.task == 'xquad' else TYDIQA_LANGUAGES
        for language in LANGUAGES:
            print(f"Evaluating on {LANGUAGES[language]}...")
            # Load the evaluation dataset
            eval_dataloader = load_eval_dataset(
                language, batch_size=1)

            # Evaluate the model
            average_exact_match, average_f1, average_percent_deleted_tokens, \
                eval_runtime = eval_loop(eval_dataloader)

            exact_match_data.append(average_exact_match)
            f1_data.append(average_f1)
            percent_deleted_tokens_data.append(
                average_percent_deleted_tokens)
            runtime_data.append(eval_runtime)
            size_data.append(len(eval_dataloader.dataset))

        # Save the evaluation metrics to a CSV file
        eval_metrics = pd.DataFrame({
            'Language': list(LANGUAGES.values()),
            'Language Code': list(LANGUAGES.keys()),
            'Eval Exact Match': exact_match_data,
            'Eval F1': f1_data,
            'Eval Percent Deleted Tokens': percent_deleted_tokens_data,
            'Eval Runtime': runtime_data,
            'Size': size_data,
        })

    # Make directory for eval results
    os.makedirs(
        f"eval_results/{args.task}/{args.model_type}", exist_ok=True)

    outfile = f"eval_results/{args.task}/{args.model_type}/{args.model_name}_seed{args.random_seed}.csv"
    eval_metrics.to_csv(outfile, index=False)

    print(f"Saved evaluation metrics to: {outfile}")
