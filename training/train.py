# train.py
# Author: Julie Kallini

import sys
sys.path.append('..')
from utils import (
    CHECKPOINT_PATH,
    DIAGNOSTIC_TASKS,
    FINETUNE_TASKS,
    LM_TASK,
    CHAR_IIT_TASKS,
    MODEL_ARCHITECTURES,
    get_task_dataset,
    load_model_from_hf,
    load_model_from_scratch,
    load_model_from_path,
)
from trainer import (
    T5Trainer,
    MrT5Trainer,
    MrT5TrainingArguments,
    DecoderBaselineT5Trainer,
    BaselineMrT5Trainer,
    BPT5Trainer,
    CanineT5Trainer,
)
from datasets import load_dataset
from data.data_collator_finetuning import XNLIDataCollator, QADataCollator
from models.modeling_mrt5 import MrT5Config
from models.modeling_t5 import T5Config
from models.modeling_bpt5 import BPT5Config
from models.modeling_canine import CanineT5Config
from transformers import AutoTokenizer
import math
import numpy as np
import torch
import argparse
import os
from accelerate import Accelerator


class PassthroughDataCollator:
    def __init__(self, pad_token_id=0):
        """
        Initialize the data collator with a specified pad token ID.

        Args:
            pad_token_id (int): The token ID used for padding. Default is 0.
        """
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        input_ids = torch.tensor([example['input_ids']
                                 for example in batch], dtype=torch.long).squeeze(1)
        labels = torch.tensor([example['labels']
                              for example in batch], dtype=torch.long).squeeze(1)

        # Replace pad token IDs in labels with -100
        labels[labels == self.pad_token_id] = -100

        if 'attention_mask' in batch[0]:
            attention_mask = torch.tensor([example['attention_mask']
                                           for example in batch], dtype=torch.long).squeeze(1)
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }

        return {
            'input_ids': input_ids,
            'labels': labels
        }

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Train a ByT5 model.')

    # Set LM task options
    LM_TASKS = [LM_TASK] + [LM_TASK + "_multilingual"]

    # Task arguments
    parser.add_argument(
        'training_task',
        choices=LM_TASKS + DIAGNOSTIC_TASKS + FINETUNE_TASKS + CHAR_IIT_TASKS,
        help='Task to train the model on.'
    )

    # Model arguments
    parser.add_argument('--model_type',
                        default='T5',
                        const='all',
                        nargs='?',
                        choices=MODEL_ARCHITECTURES,
                        help='Type of model architecture to train.')
    parser.add_argument('--model_name', type=str,
                        default='google/byt5-small', help='Model name.')
    parser.add_argument('--input_seq_length', type=int,
                        default=1024, help='Input sequence length for the model.')

    # Training arguments
    parser.add_argument('--num_train_epochs', type=int, default=1,
                        help='Number of training epochs.')
    parser.add_argument('--effective_batch_size', type=int,
                        default=64, help='Effective batch size.')
    parser.add_argument('--per_device_train_batch_size', type=int,
                        default=16, help='Batch size per device during training.')
    parser.add_argument('--per_device_eval_batch_size', type=int,
                        default=8, help='Batch size per device during evaluation.')
    parser.add_argument('--max_steps', type=int,
                        default=-1, help='Maximum number of steps.')
    parser.add_argument('--warmup_steps', type=int,
                        default=0, help='Number of warmup steps.')
    parser.add_argument('--save_steps', type=int,
                        default=500, help='Number of steps before saving a checkpoint.')
    parser.add_argument('--logging_steps', type=int,
                        default=50, help='Number of logging steps.')
    parser.add_argument('--eval_steps', type=int,
                        default=500, help='Number of eval steps.')
    parser.add_argument('--eval_accumulation_steps', type=int,
                        default=4, help='Number of eval accumulation steps.')
    parser.add_argument('--learning_rate', type=float,
                        default=0.0001, help='Learning rate.')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument('--run_name', type=str, default=None,)
    parser.add_argument('--disable_wandb', action='store_true',
                        help='Disable Weights & Biases logging.')
    parser.add_argument('--disable_compile', action='store_true',
                    help='Disable compiling torch.')
    parser.add_argument('--use_cache', action='store_true',
                        help='Use cache for training.')
    parser.add_argument('--train_from_scratch',
                        action='store_true', help='Train from scratch.')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model checkpoint to load.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint to load from model path.')
    parser.add_argument('--resume_from_checkpoint', action='store_true')
    parser.add_argument('--lr_scheduler_type', type=str, default='linear')

    # Model arguments for training from scratch
    parser.add_argument('--absolute_pos_embeds', action='store_true',
                        help='Use absolute instead of relative position embeddings.')
    parser.add_argument('--relative_attention_num_buckets', type=int,
                        default=32, help='Relative attention num buckets.')
    parser.add_argument('--relative_attention_max_distance', type=int,
                        default=128, help='Relative attention max distance.')
    parser.add_argument('--d_ff', type=int,
                        default=3584, help='Feed-forward dimensionality.')
    parser.add_argument('--d_model', type=int,
                        default=1472, help='Model dimensionality.')
    parser.add_argument('--num_heads', type=int,
                        default=6, help='Number of heads.')
    parser.add_argument('--num_encoder_layers', type=int,
                        default=12, help='Number of encoder layers.')
    parser.add_argument('--num_decoder_layers', type=int,
                        default=4, help='Number of decoder layers.')

    # MrT5 specific arguments
    parser.add_argument('--delete_gate_loss_coeff', type=float, default=0.0,
                        help='Coefficient for the delete gate loss.')
    parser.add_argument('--hard_delete_train_prob', type=float, default=0.0,
                        help='Probability of hard deletion during training.')
    parser.add_argument('--loss_function',
                        type=str,
                        default='gate_mean_loss',
                        const='gate_mean_loss',
                        nargs='?',
                        choices=['gate_mean', 'clamped_logits_mean',
                                 'gate_var', 'logits_mean',
                                 'clamped_logits_mean_with_entropy_reg',
                                 'gate_mean_with_entropy_reg',],
                        help='Loss function for delete gate.')
    parser.add_argument('--sigmoid_mask_scale', type=float,
                        default=-30.0, help='Scale of sigmoid mask.')
    parser.add_argument('--deletion_threshold', type=float,
                        default=-15.0, help='Deletion threshold.')
    parser.add_argument('--regularizer_delay', type=int,
                        help='Number of steps before applying regularizer.')
    parser.add_argument('--scores_loss_coeff', type=float, default=0.0,
                        help='Coefficient for the key/query norm loss.')
    parser.add_argument('--use_softmax1',
                        action='store_true', help='Specify layer to add delete gate after.')
    parser.add_argument('--delete_gate_layer', type=int, default=3, help='Layer to add delete gate after.')

    # Entropy regularization loss
    parser.add_argument('--entropy_reg_coeff_1', type=float, default=0.0,
                        help='First coefficient for entropy regularization.')
    parser.add_argument('--entropy_reg_coeff_2', type=float, default=0.0,
                        help='Second coefficient for entropy regularization.')
    
    # P-controller arguments
    parser.add_argument('--target_deletion_rate', type=float, default=None,)
    parser.add_argument('--controller_p', type=float, default=0.5,)
    parser.add_argument('--controller_i', type=float, default=0.00001,)
    parser.add_argument('--controller_step', type=int, default=1,)

    # RandomT5 specific arguments
    parser.add_argument('--random_deletion_probability', type=float, default=0.5,
                        help='Probability of deletion during training (RandomT5 model only).')

    # FixedT5 specific arguments
    parser.add_argument('--fixed_deletion_amount', type=float, default=0.5,
                        help='Amount of deletion for fixed deletion baseline (FixedT5 model only).')
    
    # BPT5 specific arguments
    parser.add_argument('--boundary_predictor_layer', type=int, default=3,
                        help='Layer to add boundary predictor after.')
    parser.add_argument('--boundary_predictor_type', type=str, default='gumbel',
                        help='Type of boundary predictor to use.')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Temperature for boundary predictor.')
    parser.add_argument('--prior', type=float, default=0.2)

    # CanineT5 specific arguments
    parser.add_argument('--downsampling_layer', type=int, default=3,
                        help='Layer to add downsampling after.')
    parser.add_argument('--downsampling_rate', type=int, default=4,
                        help='Downsampling rate for CanineT5 model.')

    args = parser.parse_args()

    if args.training_task in DIAGNOSTIC_TASKS:
        os.environ["WANDB_PROJECT"] = "diagnostic_mrt5"
    elif args.training_task in FINETUNE_TASKS + CHAR_IIT_TASKS:
        os.environ["WANDB_PROJECT"] = "finetune_mrt5"
    else:
        os.environ["WANDB_PROJECT"] = "mrt5"

    # Initialize the accelerator
    accelerator = Accelerator()

    # Set the random seed for reproducibility
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    if args.eval_steps % args.logging_steps != 0:
        raise ValueError(
            "logging_steps should divide eval_steps for logging to work correctly.")

    if args.train_from_scratch and args.model_path is not None:
        raise ValueError(
            "Cannot train from scratch and load from path simultaneously.")

    accelerator.print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    accelerator.print("Model type:", args.model_type)

    # Initialize the configuration
    if args.model_type in ('T5', 'DecoderBaselineT5'):
        t5_config = T5Config.from_pretrained(args.model_name)
    elif args.model_type == 'MrT5':
        t5_config = MrT5Config.from_pretrained(
            args.model_name,
            deletion_type="scaled_sigmoid",
            sigmoid_mask_scale=args.sigmoid_mask_scale,
            deletion_threshold=args.deletion_threshold,
            delete_gate_layer=args.delete_gate_layer,
        )
    elif args.model_type == 'LogSigmoidMrT5':
        t5_config = MrT5Config.from_pretrained(
            args.model_name,
            deletion_type="log_sigmoid",
            sigmoid_mask_scale=args.sigmoid_mask_scale,
            deletion_threshold=args.deletion_threshold,
            delete_gate_layer=args.delete_gate_layer,
        )
    elif args.model_type == 'RandomT5':
        t5_config = MrT5Config.from_pretrained(
            args.model_name,
            deletion_type="random",
            random_deletion_probability=args.random_deletion_probability,
            sigmoid_mask_scale=args.sigmoid_mask_scale,
            deletion_threshold=args.deletion_threshold,
            delete_gate_layer=args.delete_gate_layer,
        )
    elif args.model_type == 'FixedT5':
        t5_config = MrT5Config.from_pretrained(
            args.model_name,
            deletion_type="fixed",
            fixed_deletion_amount=args.fixed_deletion_amount,
            sigmoid_mask_scale=args.sigmoid_mask_scale,
            deletion_threshold=args.deletion_threshold,
            delete_gate_layer=args.delete_gate_layer,
        )
    elif args.model_type == 'BPT5':
        t5_config = BPT5Config.from_pretrained(
            args.model_name,
            boundary_predictor_layer=args.boundary_predictor_layer,
            boundary_predictor_type=args.boundary_predictor_type,
            temperature=args.temperature,
            prior=args.prior,
        )
    elif args.model_type == 'CanineT5':
        t5_config = CanineT5Config.from_pretrained(
            args.model_name,
            downsampling_layer=args.downsampling_layer,
            downsampling_rate=args.downsampling_rate,
        )
    else:
        raise ValueError(
            f"Model type must be one of {', '.join(MODEL_ARCHITECTURES)}.")

    # Additional config arguments
    t5_config.d_ff = args.d_ff
    t5_config.d_model = args.d_model
    t5_config.num_heads = args.num_heads
    t5_config.num_layers = args.num_encoder_layers
    t5_config.num_decoder_layers = args.num_decoder_layers
    t5_config.relative_attention_num_buckets = args.relative_attention_num_buckets
    t5_config.relative_attention_max_distance = args.relative_attention_max_distance
    t5_config.use_softmax1 = args.use_softmax1
    if args.absolute_pos_embeds:
        t5_config.has_absolute_position_embeddings = True
        t5_config.max_position_embeddings = args.input_seq_length
    else:
        t5_config.has_absolute_position_embeddings = False

    if args.train_from_scratch:
        accelerator.print("Training from scratch...")
        model = load_model_from_scratch(args.model_type, t5_config)
        # Apply layer scaling to encoder and decoder independently
        model.encoder.apply(lambda module: model._init_weights(
            module, 1 / math.sqrt(model.config.num_layers)))
        model.decoder.apply(lambda module: model._init_weights(
            module, 1 / math.sqrt(model.config.num_decoder_layers)))
    elif args.model_path is not None:
        accelerator.print("Loading model from path...")
        model = load_model_from_path(
            model_class=args.model_type,
            model_path=args.model_path,
        )
    else:
        accelerator.print("Loading model from Hugging Face...")
        model = load_model_from_hf(args.model_type, args.model_name, t5_config)

    total_params = model.num_parameters()
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)

    accelerator.print(f"Total parameters: {total_params}")
    accelerator.print(f"Trainable parameters: {trainable_params}")

    # Calculate appropriate gradient accumulation steps for effective batch size
    gradient_accumulation_steps = args.effective_batch_size // (args.per_device_train_batch_size * accelerator.num_processes)
    accelerator.print(f"Num devices: {accelerator.num_processes}")
    accelerator.print(f"Effective batch size: {args.effective_batch_size}")
    accelerator.print(f"Per-device batch size: {args.per_device_train_batch_size}")
    accelerator.print(f"Gradient accumulation steps: {gradient_accumulation_steps}")

    run_name = args.run_name if args.run_name else f"{args.model_type}_{args.model_name.split('/')[-1]}_seed{args.random_seed}"
    result_dir = f"{CHECKPOINT_PATH}/{args.training_task}/{args.model_type}/{run_name}_seed{args.random_seed}"
    checkpoint_dir = f"{result_dir}/checkpoints"
    log_dir = f"{result_dir}/logs"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Format the dataset path based on the training task
    if args.training_task in CHAR_IIT_TASKS:
        train_dataset = get_task_dataset(args.training_task, split="train", iterable_dataset=False)
        eval_dataset = get_task_dataset(args.training_task, split="validation", iterable_dataset=False)
        collator = PassthroughDataCollator(tokenizer.pad_token_id)
    elif args.training_task not in FINETUNE_TASKS:
        train_dataset = get_task_dataset(
            args.training_task, "train", iterable_dataset=True)
        eval_dataset = get_task_dataset(
            args.training_task, "validation", iterable_dataset=True)
        # Initialize the data collator
        collator = PassthroughDataCollator(tokenizer.pad_token_id)
    elif args.training_task == 'xnli':
        train_dataset = load_dataset("xnli", "en", split="train").shuffle(seed=args.random_seed)
        eval_dataset = load_dataset("xnli", "en", split="validation").shuffle(seed=args.random_seed)
        collator = XNLIDataCollator(tokenizer, max_length=1024)
    elif args.training_task == 'xquad':
        train_dataset = load_dataset("squad", split="train").shuffle(seed=args.random_seed)
        eval_dataset = load_dataset("squad", split="validation").shuffle(seed=args.random_seed)
        collator = QADataCollator(tokenizer, max_length=2048)
    elif args.training_task == 'tydiqa':
        dataset = load_dataset("tydiqa", "secondary_task", split="train").shuffle(seed=args.random_seed)
        dataset = dataset.train_test_split(test_size=0.2)
        train_dataset = dataset['train']
        eval_dataset = dataset['test']
        collator = QADataCollator(tokenizer, max_length=2048)

    # Define training arguments
    training_args = MrT5TrainingArguments(
        output_dir=checkpoint_dir,
        logging_dir=log_dir,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",
        seed=args.random_seed,
        overwrite_output_dir=True,
        load_best_model_at_end=True,
        torch_compile=(not args.disable_compile),
        report_to=[] if args.disable_wandb else "wandb",
        run_name=run_name,
        remove_unused_columns=False,
        delete_gate_loss_coeff=args.delete_gate_loss_coeff, # coefficient for delete gate loss
        loss_function=args.loss_function,                   # loss function for delete gate
        hard_delete_train_prob=args.hard_delete_train_prob, # hard deletion probability
        regularizer_delay=args.regularizer_delay,           # regularizer delay
        target_deletion_rate=args.target_deletion_rate,     # target deletion rate
        controller_p=args.controller_p,                     # Controller value
        controller_i=args.controller_i,                     # Controller value
        controller_step=args.controller_step,               # P-controller step
        scores_loss_coeff=args.scores_loss_coeff,           # coefficient for scores loss   
        entropy_reg_coeff_1=args.entropy_reg_coeff_1,       # coefficient for entropy regularization
        entropy_reg_coeff_2=args.entropy_reg_coeff_2,       # coefficient for entropy regularization
    )

    # Initialize the Trainer
    if args.model_type in ('MrT5', 'LogSigmoidMrT5'):
        Trainer = MrT5Trainer
    elif args.model_type in ('RandomT5', 'FixedT5'):
        Trainer = BaselineMrT5Trainer
    elif args.model_type == 'DecoderBaselineT5':
        Trainer = DecoderBaselineT5Trainer
    elif args.model_type == 'T5':
        Trainer = T5Trainer
    elif args.model_type == 'BPT5':
        Trainer = BPT5Trainer
    elif args.model_type == 'CanineT5':
        Trainer = CanineT5Trainer
    else:
        raise ValueError(
            f"Model type must be one of {', '.join(MODEL_ARCHITECTURES)}.")

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    
    accelerator.print("Training the model...")
    # Train the model
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    