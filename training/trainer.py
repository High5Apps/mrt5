import sys
sys.path.append('..')

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from dataclasses import dataclass, field
from typing import Optional
import time
import torch

@dataclass
class MrT5TrainingArguments(Seq2SeqTrainingArguments):
    delete_gate_loss_coeff: float = field(
        default=None, metadata={"help": "Coefficient for the delete gate loss."}
    )
    target_deletion_rate: Optional[float] = field(
        default=None, metadata={"help": "Target deletion rate for the delete gate."}    
    )
    controller_p: float = field(
        default=0.5, metadata={"help": "Proportional gain for the controller."} 
    )
    controller_i: float = field(
        default=1e-5, metadata={"help": "Integral gain for the controller."}
    )
    controller_d: float = field(
        default=1e-6, metadata={"help": "Differential gain for the controller."}
    )
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
        if args.delete_gate_loss_coeff is None and args.target_deletion_rate is None:
            raise ValueError("One or more of the following args must be set: delete_gate_loss_coeff, target_deletion_rate")

        self.delete_gate_loss_coeff = torch.tensor(args.delete_gate_loss_coeff or 0.0)

        # Controller parameters
        self.i_acc = 0.0
        self.previous_err = 0.0
        self._microstep = 0 # This must be tracked to know when to skip PID controller updates when gradient_accumulation_steps > 1

        print('Setting requires_grad = False for delete gate bias')
        self.delete_gate_bias = model.encoder.block[3].delete_gate.feed_forward.bias
        self.delete_gate_bias.requires_grad = False
        self.initial_delete_gate_bias = self.delete_gate_bias.item()

    def pid_controller(self, target_deletion, current_deletion):
        err = target_deletion - current_deletion
        self.i_acc += err

        p = self.args.controller_p * err
        i = self.i_acc * self.args.controller_i
        d = self.args.controller_d * (err - self.previous_err)

        self.previous_err = err

        return p + i + d

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
        delete_gate_output = outputs.delete_gate_output.squeeze(-1).detach()
        non_pad_mask = input_ids != 0 # Create a mask to exclude PAD tokens

        # Log metrics
        num_non_pad_tokens = non_pad_mask.sum()
        num_non_pad_deleted_tokens = ((delete_gate_output < model.config.deletion_threshold) & non_pad_mask).sum()
        percent_non_pad_deleted_tokens = num_non_pad_deleted_tokens / num_non_pad_tokens * 100
        self.log({
            'percent_non_pad_deleted_tokens': percent_non_pad_deleted_tokens.item(),
            'delete_gate_bias': self.delete_gate_bias.item(),
            'delete_gate_mean': delete_gate_output[non_pad_mask].mean().item(),
            'delete_gate_output': delete_gate_output[non_pad_mask],
        }, time.time())

        # Update delete gate bias if needed
        self._microstep += 1
        if self.args.target_deletion_rate is not None and self._microstep % self.args.gradient_accumulation_steps == 0:
            self.delete_gate_bias.data = self.initial_delete_gate_bias - self.pid_controller(
                self.args.target_deletion_rate,
                percent_non_pad_deleted_tokens / 100)

        return (loss, outputs) if return_outputs else loss
