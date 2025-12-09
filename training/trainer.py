import sys
sys.path.append('..')

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from dataclasses import dataclass, field
from typing import Optional
import time

@dataclass
class MrT5TrainingArguments(Seq2SeqTrainingArguments):
    hard_delete: bool = field(
        default=False, metadata={"help": "Whether to actually remove hidden states from computation to realize efficiency gains."}
    )

class MrT5Trainer(Seq2SeqTrainer):
    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        model_init=None,
        compute_loss_func=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self.model_accepts_loss_kwargs = False
        input_ids = inputs.pop("input_ids")
        attention_mask = inputs.pop("attention_mask") if "attention_mask" in inputs else None
        labels = inputs.pop("labels")

        outputs = model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
            hard_delete=self.args.hard_delete,
        )

        # Compute the cross entropy loss
        loss = outputs.loss

        # Compute the delete gate loss
        delete_gate_output = outputs.delete_gate_output.squeeze(-1)
        non_pad_mask = input_ids != 0 # Create a mask to exclude PAD tokens

        # Log metrics
        num_non_pad_tokens = non_pad_mask.sum()
        deletion_threshold = model.encoder.block[model.config.delete_gate_layer].deletion_threshold
        num_non_pad_deleted_tokens = ((delete_gate_output <= deletion_threshold) & non_pad_mask).sum()
        percent_non_pad_deleted_tokens = num_non_pad_deleted_tokens / num_non_pad_tokens * 100
        self.log({
            'percent_non_pad_deleted_tokens': percent_non_pad_deleted_tokens.item(),
            'delete_gate_mean': delete_gate_output[non_pad_mask].mean().item(),
            'delete_gate_output': delete_gate_output[non_pad_mask],
            'deletion_threshold': deletion_threshold.item(),
        }, time.time())

        return (loss, outputs) if return_outputs else loss
