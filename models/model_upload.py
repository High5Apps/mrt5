# upload_mrt5.py
# Author: Julie Kallini
# Uploads custom MrT5 model + config + code to Hugging Face Hub
# model_path = "/nlp/scr3/nlp/llms-in-llms/mrt5/models/span_corruption_multilingual/MrT5/mrt5_span_corruption_multilingual_50%_seed40/checkpoints/checkpoint-5000/"
# model_path = "/nlp/scr3/nlp/llms-in-llms/mrt5/models/span_corruption_multilingual/MrT5/mrt5-large_span_corruption_multilingual_50%_seed23/checkpoints/checkpoint-5000/"

import sys
sys.path.append('..')

from huggingface_hub import create_repo, upload_file
from modeling_mrt5 import MrT5ForConditionalGeneration
# from configuration_mrt5 import MrT5Config
import os
import argparse

def push_custom_files(repo_id, local_dir, files_to_upload):
    for filename in files_to_upload:
        path = os.path.join(local_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found.")
        upload_file(
            path_or_fileobj=path,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="model"
        )
        print(f"Uploaded {filename} to {repo_id}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload MrT5 to Hugging Face Hub")
    parser.add_argument("--model-dir", required=True, help="Path to trained model checkpoint dir")
    parser.add_argument("--repo", required=True, help="Hugging Face repo name (e.g. juliekallini/mrt5-custom)")
    args = parser.parse_args()

    model_dir = args.model_dir
    repo_id = args.repo

    # Load model
    print("Loading model...")
    model = MrT5ForConditionalGeneration.from_pretrained(model_dir)

    # Update and save config with custom info
    config = model.config
    config.model_type = "mrt5"
    config.architectures = ["MrT5ForConditionalGeneration"]
    config.auto_map = {
        "AutoConfig": "configuration_mrt5.MrT5Config",
        "AutoModelForSeq2SeqLM": "modeling_mrt5.MrT5ForConditionalGeneration"
    }
    config.save_pretrained(model_dir)

    # Create repo (if needed)
    create_repo(repo_id, exist_ok=True)

    # Push model
    print("Pushing model...")
    model.push_to_hub(repo_id)

    # Push custom code files
    print("Uploading custom code...")
    push_custom_files(
        repo_id=repo_id,
        local_dir=".",  # assuming you're running this from the dir containing the files
        files_to_upload=["modeling_t5.py", "modeling_mrt5.py", "configuration_mrt5.py"]
    )

    print("✅ Upload complete!")


