# trainer.py
# Author: Julie Kallini

import sys
sys.path.append('..')

import statistics
import torch
import nltk
import numpy as np
from transformers import TrainingArguments, Trainer
from dataclasses import dataclass, field
from typing import Optional
from utils import calculate_seq_accuracy, calculate_token_accuracy


@dataclass
class MrT5TrainingArguments(TrainingArguments):
    delete_gate_loss_coeff: float = field(
        default=0.0, metadata={"help": "Coefficient for the delete gate loss."}
    )
    loss_function: str = field(
        default="gate_mean_loss", metadata={"help": "Loss function to use for delete gate training."}
    )
    hard_delete_train_prob: float = field(
        default=0.0, metadata={"help": "Probability of hard deletion during training."}
    )
    regularizer_delay: Optional[int] = field(
        default=None, metadata={"help": "Number of steps to delay regularizer."}    
    )
    target_deletion_rate: Optional[float] = field(
        default=None, metadata={"help": "Target deletion rate for the delete gate."}    
    )
    controller_p: float = field(
        default=0.5, metadata={"help": "Proportional gain for the controller."} 
    )
    controller_i: float = field(
        default=0.00001, metadata={"help": "Integral gain for the controller."}
    )
    controller_step: int = field(
        default=1, metadata={"help": "Number of steps between controller updates."}
    )
    scores_loss_coeff: float = field(
        default=0.0, metadata={"help": "Coefficient for the key/query norm loss."}
    )
    entropy_reg_coeff_1: float = field(
        default=0.0, metadata={"help": "First coefficient for the entropy regularization loss."}
    )
    entropy_reg_coeff_2: float = field(
        default=0.0, metadata={"help": "Second coefficient for the entropy regularization loss."}
    )


class T5Trainer(Trainer):
    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        include_edit_distance=False,
        random_seed=None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        # Initialize the metrics to log
        self.include_edit_distance = include_edit_distance
        self.metrics = self.init_metrics()
        self.rng = np.random.default_rng(random_seed)

    def init_metrics(self):
        metrics = {
            "cross_entropy_loss": [],
            "prediction_seq_accuracy": [],
            "prediction_token_accuracy": [],
            "eval_cross_entropy_loss": [],
            "eval_prediction_seq_accuracy": [],
            "eval_prediction_token_accuracy": [],
        }
        if self.include_edit_distance:
            metrics["eval_edit_distance"] = []
        return metrics

    def calculate_edit_distance(self, labels, outputs):
        logits = outputs.logits
        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_texts = [self.tokenizer.decode(
            pred, skip_special_tokens=True) for pred in predicted_ids]
        target_texts = [self.tokenizer.decode(
            target, skip_special_tokens=True) for target in labels]
        edit_distances = [nltk.edit_distance(
            ref, pred) for ref, pred in zip(target_texts, predicted_texts)]
        average_edit_distance = sum(edit_distances) / len(edit_distances)
        return average_edit_distance

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.pop("input_ids")
        attention_mask = inputs.pop(
            "attention_mask") if "attention_mask" in inputs else None
        labels = inputs.pop("labels")
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                        labels=labels, output_hidden_states=True)

        loss = outputs.loss
        prediction_seq_accuracy = calculate_seq_accuracy(labels, outputs)
        prediction_token_accuracy = calculate_token_accuracy(
            labels, outputs)

        # Flag for logging training vs. evaluation metrics
        eval_flag = "" if model.training else "eval_"
        self.metrics[f"{eval_flag}cross_entropy_loss"].append(
            loss.detach().item())
        self.metrics[f"{eval_flag}prediction_seq_accuracy"].append(
            prediction_seq_accuracy)
        self.metrics[f"{eval_flag}prediction_token_accuracy"].append(
            prediction_token_accuracy)

        # Compute the edit distance
        if not model.training and self.include_edit_distance:
            self.metrics[f"eval_edit_distance"].append(
                self.calculate_edit_distance(labels, outputs))

        return (loss, outputs) if return_outputs else loss

    def log(self, logs):
        # Format the metrics to log
        formatted_metrics = {k: round(statistics.fmean(v), 4)
                             for k, v in self.metrics.items() if "histogram" not in k and len(v) > 0}
        logs.update(formatted_metrics)

        # Reinitalize the metrics
        self.metrics = self.init_metrics()

        # Call the superclass method to handle standard logging
        super().log(logs)


class MrT5Trainer(T5Trainer):
    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        include_edit_distance=False,
        random_seed=None,
    ):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
            include_edit_distance,
            random_seed,
        )
        self.loss_function = args.loss_function
        self.hard_delete_train_prob = args.hard_delete_train_prob
        self.deletion_threshold = model.config.deletion_threshold
        self.regularizer_delay = args.regularizer_delay
        self.delete_gate_loss_coeff = args.delete_gate_loss_coeff
        self.scores_loss_coeff = args.scores_loss_coeff
        self.target_deletion_rate = args.target_deletion_rate
        self.delete_gate_layer = model.config.delete_gate_layer

        # Controller parameters
        self.i = args.controller_i
        self.p = args.controller_p
        self.controller_step = args.controller_step
        self.p_acc = 0.0
        self.i_acc = 0.0

    def init_metrics(self):
        metrics = {
            "cross_entropy_loss": [],
            "delete_gate_loss": [],
            "total_loss": [],
            "prediction_seq_accuracy": [],
            "prediction_token_accuracy": [],
            "delete_gate_average": [],
            "delete_gate_std": [],
            "delete_gate_max_value": [],
            "delete_gate_min_value": [],
            "percent_deleted_tokens": [],
            "percent_non_pad_deleted_tokens": [],
            "delete_gate_loss_coeff": [],
            "new_seq_len": [],
            "value_counts": [],
            "delete_gate_histogram": [],
            "eval_hard_cross_entropy_loss": [],
            "eval_hard_total_loss": [],
            "eval_hard_prediction_seq_accuracy": [],
            "eval_hard_prediction_token_accuracy": [],
            "eval_soft_cross_entropy_loss": [],
            "eval_soft_total_loss": [],
            "eval_soft_prediction_seq_accuracy": [],
            "eval_soft_prediction_token_accuracy": [],
            "eval_delete_gate_loss": [],
            "eval_delete_gate_average": [],
            "eval_delete_gate_std": [],
            "eval_delete_gate_max_value": [],
            "eval_delete_gate_min_value": [],
            "eval_percent_deleted_tokens": [],
            "eval_percent_non_pad_deleted_tokens": [],
            "eval_delete_gate_loss_coeff": [],
            "eval_new_seq_len": [],
        }
        if self.include_edit_distance:
            metrics["eval_soft_edit_distance"] = []
            metrics["eval_hard_edit_distance"] = []
        return metrics

    def compute_entropy_reg_loss(self, gate_output):
        def get_gate_entropy(g):
            gexp = g.exp()
            H = gexp * g + \
                (1 - gexp) * torch.log1p(-gexp.clamp(max=1-torch.finfo(gexp.dtype).eps))
            return -H

        H_1 = get_gate_entropy(gate_output).mean(dim=1)
        H_2 = get_gate_entropy(
            torch.logsumexp(gate_output, dim=1) - torch.log(torch.tensor(gate_output.shape[1])))

        return H_1, H_2

    def i_controller(self, target_deletion, current_deletion, curr_coeff):
        # Calculate error
        error = target_deletion - current_deletion
        return max(0.0, curr_coeff + self.i * error)

    def pi_controller(self, target_deletion, current_deletion):
        err = target_deletion - current_deletion
        self.p_acc = 0.9 * self.p_acc + 0.1 * self.p * err
        self.i_acc = self.i_acc + self.i * err
        return max(0.0, self.p_acc + self.i_acc)
    
    def __get_clamped_mean(self, tensor, min_value=5.0):
        return (tensor.clamp(min_value).mean()-min_value).mean()

    def __scores_loss(self, outputs):
        means = []
        for s in outputs.encoder_scores[self.delete_gate_layer:]:
            means.append(self.__get_clamped_mean(s))
        for s in outputs.cross_attention_scores:
            means.append(self.__get_clamped_mean(s))    
        return sum(means) / len(means)

    def __compute_loss(self, outputs, input_ids, log_deletion_metrics=False, metrics_prefix=""):

        # Get delete gate mask
        delete_gate_output = outputs.delete_gate_output.squeeze(-1)
        delete_gate_logits = outputs.delete_gate_logits.squeeze(-1)

        # Create a mask to exclude PAD tokens
        non_pad_mask = input_ids != 0

        # Compute the delete gate loss, excluding PAD tokens
        if "gate_mean" in self.loss_function:
            delete_gate_loss = delete_gate_output[non_pad_mask].mean()
        elif "clamped_logits_mean" in self.loss_function:
            delete_gate_loss = torch.clamp(
                delete_gate_logits[non_pad_mask], min=self.deletion_threshold * 2).mean()
        elif self.loss_function == "logits_mean":
            delete_gate_loss = delete_gate_logits[non_pad_mask].mean()
        elif self.loss_function == "gate_var_loss":
            delete_gate_loss = - \
                delete_gate_output[non_pad_mask].var(dim=0).mean()
        else:
            raise ValueError("Invalid loss function.")
        delete_gate_loss *= self.delete_gate_loss_coeff

        if "entropy_reg" in self.loss_function:
            H_1, H_2 = self.compute_entropy_reg_loss(
                delete_gate_output)
            mean_H_1 = H_1.mean()
            mean_H_2 = H_2.mean()
            delete_gate_loss += \
                self.args.entropy_reg_coeff_1 * mean_H_1 - \
                self.args.entropy_reg_coeff_2 * mean_H_2

        # Compute the cross entropy loss
        cross_entropy_loss = outputs.loss

        # Count on average how many tokens are deleted
        batch_size, seq_len = input_ids.shape[0:2]
        num_deleted_tokens = (delete_gate_output <
                              self.deletion_threshold).sum()
        percent_deleted_tokens = num_deleted_tokens / \
            (batch_size * seq_len) * 100

        # Count on average how many tokens are deleted, excluding pad tokens
        batch_size, seq_len = input_ids.shape[0:2]
        non_pad_mask = input_ids != 0
        num_non_pad_tokens = non_pad_mask.sum()
        num_deleted_tokens = (
            (delete_gate_output < self.deletion_threshold) & non_pad_mask).sum()
        percent_non_pad_deleted_tokens = num_deleted_tokens / num_non_pad_tokens * 100

        if self.regularizer_delay is not None and self.state.global_step < self.regularizer_delay:
            loss = cross_entropy_loss
        else:
            # Adjust delete gate loss coefficient based on percentage of tokens deleted
            if self.target_deletion_rate is not None and self.state.global_step % self.controller_step == 0:
                self.delete_gate_loss_coeff = self.pi_controller(
                    self.target_deletion_rate, percent_non_pad_deleted_tokens / 100)
            loss = cross_entropy_loss + delete_gate_loss

        if self.scores_loss_coeff > 0:
            loss = loss + self.__scores_loss(outputs) * self.scores_loss_coeff

        # Update running metrics
        if log_deletion_metrics:
            self.metrics[f"{metrics_prefix}delete_gate_loss"].append(
                delete_gate_loss.detach().item())
            self.metrics[f"{metrics_prefix}delete_gate_average"].append(
                delete_gate_output.mean().detach().item())
            self.metrics[f"{metrics_prefix}delete_gate_std"].append(
                delete_gate_output.std(dim=1).mean().detach().item())
            self.metrics[f"{metrics_prefix}delete_gate_max_value"].append(
                delete_gate_output.max(dim=1).values.mean().detach().item())
            self.metrics[f"{metrics_prefix}delete_gate_min_value"].append(
                delete_gate_output.min(dim=1).values.mean().detach().item())
            self.metrics[f"{metrics_prefix}percent_deleted_tokens"].append(
                percent_deleted_tokens.mean().item())
            self.metrics[f"{metrics_prefix}percent_non_pad_deleted_tokens"].append(
                percent_non_pad_deleted_tokens.mean().item())
            self.metrics[f"{metrics_prefix}new_seq_len"].append(
                outputs.encoder_last_hidden_state.shape[1])
            self.metrics[f"{metrics_prefix}delete_gate_loss_coeff"].append(
                self.delete_gate_loss_coeff)

        return loss, cross_entropy_loss, delete_gate_output

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs.pop("input_ids")
        attention_mask = inputs.pop(
            "attention_mask") if "attention_mask" in inputs else None
        labels = inputs.pop("labels")
        if model.training:
            # Log training metrics
            outputs = model(input_ids=input_ids, labels=labels,
                            attention_mask=attention_mask,
                            output_hidden_states=True,
                            output_attentions=True,
                            hard_delete=self.rng.random() < self.hard_delete_train_prob)
            loss, cross_entropy_loss, delete_gate_output = self.__compute_loss(
                outputs, input_ids, log_deletion_metrics=True)
            self.metrics["cross_entropy_loss"].append(
                cross_entropy_loss.detach().item())
            self.metrics["total_loss"].append(
                loss.detach().item())
            self.metrics["delete_gate_histogram"].append(
                delete_gate_output.detach().cpu().numpy().flatten().tolist())

            # Log prediction accuracy
            self.metrics["prediction_seq_accuracy"].append(
                calculate_seq_accuracy(labels, outputs))
            self.metrics["prediction_token_accuracy"].append(
                calculate_token_accuracy(labels, outputs))
        else:
            # Log losses for soft deletion
            outputs = model(input_ids=input_ids, labels=labels,
                            attention_mask=attention_mask,
                            output_hidden_states=True,
                            output_attentions=True,
                            hard_delete=False)
            loss, cross_entropy_loss, _ = self.__compute_loss(
                outputs, input_ids, metrics_prefix="eval_")
            self.metrics["eval_soft_cross_entropy_loss"].append(
                cross_entropy_loss.detach().item())
            self.metrics["eval_soft_total_loss"].append(
                loss.detach().item())

            # Log prediction accuracy
            self.metrics["eval_soft_prediction_seq_accuracy"].append(
                calculate_seq_accuracy(labels, outputs))
            self.metrics["eval_soft_prediction_token_accuracy"].append(
                calculate_token_accuracy(labels, outputs))

            if self.include_edit_distance:
                self.metrics["eval_soft_edit_distance"].append(
                    self.calculate_edit_distance(labels, outputs))

            # Log losses for hard deletion + other eval metrics
            outputs = model(input_ids=input_ids, labels=labels,
                            attention_mask=attention_mask,
                            output_attentions=True,
                            output_hidden_states=True, hard_delete=True)
            loss, cross_entropy_loss, _ = self.__compute_loss(
                outputs, input_ids, log_deletion_metrics=True, metrics_prefix="eval_")
            self.metrics[f"eval_hard_cross_entropy_loss"].append(
                cross_entropy_loss.detach().item())
            self.metrics[f"eval_hard_total_loss"].append(
                loss.detach().item())

            # Log prediction accuracy
            self.metrics["eval_hard_prediction_seq_accuracy"].append(
                calculate_seq_accuracy(labels, outputs))
            self.metrics["eval_hard_prediction_token_accuracy"].append(
                calculate_token_accuracy(labels, outputs))

            if self.include_edit_distance:
                self.metrics[f"eval_hard_edit_distance"].append(
                    self.calculate_edit_distance(labels, outputs))

        return (loss, outputs) if return_outputs else loss


class BaselineMrT5Trainer(T5Trainer):
    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        loss_function="gate_mean_loss",
    ):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )
        self.loss_function = loss_function
        self.deletion_threshold = model.config.deletion_threshold

    def init_metrics(self):
        return {
            "cross_entropy_loss": [],
            "delete_gate_loss": [],
            "total_loss": [],
            "delete_gate_average": [],
            "delete_gate_std": [],
            "percent_deleted_tokens": [],
            "percent_non_pad_deleted_tokens": [],
            "new_seq_len": [],
            "eval_cross_entropy_loss": [],
            "eval_delete_gate_loss": [],
            "eval_total_loss": [],
            "eval_delete_gate_average": [],
            "eval_delete_gate_std": [],
            "eval_percent_deleted_tokens": [],
            "eval_percent_non_pad_deleted_tokens": [],
            "eval_new_seq_len": [],
        }

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.pop("input_ids")
        attention_mask = inputs.pop(
            "attention_mask") if "attention_mask" in inputs else None
        labels = inputs.pop("labels")
        outputs = model(input_ids=input_ids, labels=labels,
                        attention_mask=attention_mask,
                        output_hidden_states=True)

        # Flag for logging training vs. evaluation metrics
        eval_flag = "" if model.training else "eval_"

        # Get delete gate mask
        delete_gate_output = outputs.delete_gate_output.squeeze(-1)

        # Compute the cross entropy loss
        cross_entropy_loss = outputs.loss

        # Count on average how many tokens are deleted
        batch_size, seq_len = input_ids.shape[0:2]
        num_deleted_tokens = (delete_gate_output <
                              model.config.sigmoid_mask_scale / 2).sum()
        percent_deleted_tokens = num_deleted_tokens / \
            (batch_size * seq_len) * 100

        # Count on average how many tokens are deleted, excluding pad tokens
        batch_size, seq_len = input_ids.shape[0:2]
        non_pad_mask = input_ids != 0
        num_non_pad_tokens = non_pad_mask.sum()
        num_deleted_tokens = (
            (delete_gate_output < self.deletion_threshold) & non_pad_mask).sum()
        percent_non_pad_deleted_tokens = num_deleted_tokens / num_non_pad_tokens * 100

        # Total loss
        loss = cross_entropy_loss

        # Update running metrics
        self.metrics[f"{eval_flag}cross_entropy_loss"].append(
            cross_entropy_loss.detach().item())
        self.metrics[f"{eval_flag}total_loss"].append(
            loss.detach().item())
        self.metrics[f"{eval_flag}delete_gate_average"].append(
            delete_gate_output.mean().detach().item())
        self.metrics[f"{eval_flag}delete_gate_std"].append(
            delete_gate_output.std(dim=1).mean().detach().item())
        self.metrics[f"{eval_flag}percent_deleted_tokens"].append(
            percent_deleted_tokens.mean().item())
        self.metrics[f"{eval_flag}percent_non_pad_deleted_tokens"].append(
            percent_non_pad_deleted_tokens.mean().item())
        self.metrics[f"{eval_flag}new_seq_len"].append(
            outputs.encoder_last_hidden_state.shape[1])

        return (loss, outputs) if return_outputs else loss


class DecoderBaselineT5Trainer(T5Trainer):
    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        include_edit_distance=False,
    ):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
            include_edit_distance,
        )

    def init_metrics(self):
        return {
            "cross_entropy_loss": [],
            "eval_cross_entropy_loss": [],
        }

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.pop("input_ids")
        labels = inputs.pop("labels")

        batch_size, _ = input_ids.shape

        # Create a dummy input_ids tensor to pass to the model
        input_ids = torch.tensor([[1]]*batch_size).to(input_ids.device)
        outputs = model(input_ids=input_ids, labels=labels,
                        output_hidden_states=True)

        loss = outputs.loss

        # Flag for logging training vs. evaluation metrics
        eval_flag = "" if model.training else "eval_"
        self.metrics[f"{eval_flag}cross_entropy_loss"].append(
            loss.detach().item())

        return (loss, outputs) if return_outputs else loss


class BPT5Trainer(T5Trainer):
    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        include_edit_distance=False,
        random_seed=None,
    ):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )

    def init_metrics(self):
        metrics = {
            "cross_entropy_loss": [],
            "boundaries_loss": [],
            "prediction_seq_accuracy": [],
            "prediction_token_accuracy": [],
            "percent_deleted_tokens": [],
            "new_seq_len": [],
            "eval_cross_entropy_loss": [],
            "eval_boundaries_loss": [],
            "eval_prediction_seq_accuracy": [],
            "eval_prediction_token_accuracy": [],
            "eval_percent_deleted_tokens": [],
            "eval_new_seq_len": [],
        }
        return metrics

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.pop("input_ids")
        attention_mask = inputs.pop(
            "attention_mask") if "attention_mask" in inputs else None
        labels = inputs.pop("labels")
        outputs = model(input_ids=input_ids, labels=labels,
                        attention_mask=attention_mask,
                        output_hidden_states=True)

        loss = outputs.loss
        loss_boundaries = outputs.loss_boundaries
        prediction_seq_accuracy = calculate_seq_accuracy(labels, outputs)
        prediction_token_accuracy = calculate_token_accuracy(
            labels, outputs)

        # Count on average how many tokens are deleted
        hard_boundaries = outputs.hard_boundaries
        batch_size, seq_len = input_ids.shape[0:2]
        num_deleted_tokens = torch.sum(hard_boundaries < 1.0).item()
        percent_deleted_tokens = num_deleted_tokens / \
            (batch_size * seq_len) * 100

        # Flag for logging training vs. evaluation metrics
        eval_flag = "" if model.training else "eval_"
        self.metrics[f"{eval_flag}cross_entropy_loss"].append(
            loss.detach().item())
        self.metrics[f"{eval_flag}boundaries_loss"].append(
            loss_boundaries.detach().item())
        self.metrics[f"{eval_flag}prediction_seq_accuracy"].append(
            prediction_seq_accuracy)
        self.metrics[f"{eval_flag}prediction_token_accuracy"].append(
            prediction_token_accuracy)
        self.metrics[f"{eval_flag}new_seq_len"].append(
            outputs.encoder_last_hidden_state.shape[1])
        self.metrics[f"{eval_flag}percent_deleted_tokens"].append(
            percent_deleted_tokens)

        # Compute the edit distance
        if not model.training and self.include_edit_distance:
            self.metrics[f"eval_edit_distance"].append(
                self.calculate_edit_distance(labels, outputs))

        loss = loss + loss_boundaries
        return (loss, outputs) if return_outputs else loss


class CanineT5Trainer(T5Trainer):
    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        include_edit_distance=False,
        random_seed=None,
    ):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )

    def init_metrics(self):
        metrics = {
            "cross_entropy_loss": [],
            "prediction_seq_accuracy": [],
            "prediction_token_accuracy": [],
            "percent_deleted_tokens": [],
            "new_seq_len": [],
            "eval_cross_entropy_loss": [],
            "eval_prediction_seq_accuracy": [],
            "eval_prediction_token_accuracy": [],
            "eval_percent_deleted_tokens": [],
            "eval_new_seq_len": [],
        }
        return metrics

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.pop("input_ids")
        attention_mask = inputs.pop(
            "attention_mask") if "attention_mask" in inputs else None
        labels = inputs.pop("labels")
        outputs = model(input_ids=input_ids, labels=labels,
                        attention_mask=attention_mask,
                        output_hidden_states=True)

        loss = outputs.loss
        prediction_seq_accuracy = calculate_seq_accuracy(labels, outputs)
        prediction_token_accuracy = calculate_token_accuracy(
            labels, outputs)

        # How many tokens are deleted is just determined by downsampling
        # rate, so we can just calculate it using the last encoder hidden state
        _, seq_len = input_ids.shape[0:2]
        percent_deleted_tokens = outputs.encoder_last_hidden_state.shape[1] / \
            seq_len * 100

        # Flag for logging training vs. evaluation metrics
        eval_flag = "" if model.training else "eval_"
        self.metrics[f"{eval_flag}cross_entropy_loss"].append(
            loss.detach().item())
        self.metrics[f"{eval_flag}prediction_seq_accuracy"].append(
            prediction_seq_accuracy)
        self.metrics[f"{eval_flag}prediction_token_accuracy"].append(
            prediction_token_accuracy)
        self.metrics[f"{eval_flag}new_seq_len"].append(
            outputs.encoder_last_hidden_state.shape[1])
        self.metrics[f"{eval_flag}percent_deleted_tokens"].append(
            percent_deleted_tokens)

        # Compute the edit distance
        if not model.training and self.include_edit_distance:
            self.metrics[f"eval_edit_distance"].append(
                self.calculate_edit_distance(labels, outputs))

        return (loss, outputs) if return_outputs else loss

