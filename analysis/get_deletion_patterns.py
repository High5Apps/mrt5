# get_deletion_patterns.py
# Author: Julie Kallini

import sys
sys.path.append('..')

import os
import torch
import argparse
import json
from transformers import AutoTokenizer
from utils import get_task_dataset, load_model_from_path
from tqdm import tqdm

def main(args):
    # Set device (passed as an argument or auto-detected)
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

    # Load the evaluation dataset
    eval_dataset = get_task_dataset(args.task, "test", language=args.language)

    # Randomly shuffle the dataset
    eval_dataset = eval_dataset.shuffle(seed=args.seed)

    # Load the deletion model
    del_model = load_model_from_path(
        model_class=args.model_class,
        model_name=args.model_name,
        training_task=args.task,
        seed=args.seed,
        ckpt=args.ckpt,
    ).to(device)

    # Initialize decoder input ids
    decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]]).to(device)

    # Create a list to store results
    results = []

    # Iterate through the evaluation dataset
    i = 0
    for example in tqdm(eval_dataset, total=args.sample_size):
        # Move input tokens to the appropriate device
        tokens = torch.tensor(example["input_ids"]).to(device)

        # Generate the model output
        output = del_model(input_ids=tokens, decoder_input_ids=decoder_input_ids, output_hidden_states=True)

        # Apply the delete gate threshold
        is_deleted_list = (output["delete_gate_output"] < args.deletion_threshold).squeeze().tolist()

        # Decode the input tokens
        decoded_token_list = [tokenizer.decode([t]) for t in tokens.squeeze().tolist()]

        # Append the results as a dictionary
        results.append({
            "deletion_mask": is_deleted_list,
            "decoded_input_ids": decoded_token_list,
        })

        if i >= args.sample_size:
            break
        i += 1


    # Write the results to a JSON file
    output_path = "deletion_patterns/{}_{}/".format(args.task, args.language)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_file = output_path + "{}_seed{}_ckpt{}.json".format(args.model_name, args.seed, args.ckpt)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DelT5 span corruption model on a dataset and save results to a JSON file.")
    
    # Add arguments to the parser
    parser.add_argument('model_name', type=str, help="Name of the model to load")
    parser.add_argument('--model_class', type=str, default="MrT5", help="Model class to load")
    parser.add_argument('--task', type=str, default="span_corruption", help="Task name for dataset loading")
    parser.add_argument('--language', type=str, default="en", help="Language of the dataset")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for model loading")
    parser.add_argument('--ckpt', type=int, default=3000, help="Checkpoint number for loading the model")
    parser.add_argument('--deletion_threshold', type=float, default=-15.0, help="Deletion gate threshold")
    parser.add_argument('--sample_size', type=int, default=1000, help="Number of samples to run")

    # Parse the arguments
    args = parser.parse_args()
    
    # Run the main function
    main(args)
