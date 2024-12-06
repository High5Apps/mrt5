# trainer.py
# Author: Julie Kallini

import statistics
import torch
import wandb
import nltk
import numpy as np
from transformers import TrainingArguments, Trainer
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MrT5TrainingArguments(TrainingArguments):
    delete_gate_loss_coeff: float = field(
        default=1.0, metadata={"help": "Coefficient for the delete gate loss."}
    )
    entropy_reg_coeff_1: Optional[float] = field(
        default=None, metadata={"help": "First coefficient for the entropy regularization loss."}
    )
    entropy_reg_coeff_2: Optional[float] = field(
        default=None, metadata={"help": "Second coefficient for the entropy regularization loss."}
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

    def calculate_seq_accuracy(self, labels, outputs):
        logits = outputs.logits
        # Get the predicted IDs
        predicted_ids = torch.argmax(logits, dim=-1)

        # Don't count indices of pad tokens as incorrect predictions
        pad_tokens = labels <= 0
        correct_tokens = (predicted_ids == labels) | pad_tokens

        # Compare predicted_ids with the true labels
        correct_predictions = correct_tokens.all(dim=-1).sum().item()
        total_predictions = labels.shape[0]

        # Calculate accuracy
        accuracy = correct_predictions / total_predictions

        return accuracy

    def calculate_token_accuracy(self, labels, outputs):
        logits = outputs.logits
        # Get the predicted IDs
        predicted_ids = torch.argmax(logits, dim=-1)

        # Don't count indices of pad tokens as incorrect predictions
        pad_tokens = labels <= 0
        num_pad_tokens = pad_tokens.sum().item()

        # Compare predicted_ids with the true labels
        correct_tokens = (predicted_ids == labels) | pad_tokens

        # Compare predicted_ids with the true labels
        correct_predictions = correct_tokens.sum().item() - num_pad_tokens

        # Calculate accuracy
        total_predictions = labels.numel() - num_pad_tokens
        accuracy = correct_predictions / total_predictions

        return accuracy


    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.pop("input_ids")
        attention_mask = inputs.pop(
            "attention_mask") if "attention_mask" in inputs else None
        labels = inputs.pop("labels")
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                        labels=labels, output_hidden_states=True)

        loss = outputs.loss
        prediction_seq_accuracy = self.calculate_seq_accuracy(labels, outputs)
        prediction_token_accuracy = self.calculate_token_accuracy(
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
        # Log the delete gate histogram
        # if self.model.training and wandb.run is not None \
        #         and "delete_gate_histogram" in self.metrics:
        #     wandb.log({f'delete_gate_output_hist': wandb.Histogram(
        #         self.metrics["delete_gate_histogram"], num_bins=10)}, step=self.state.global_step+1)

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
        loss_function="gate_mean_loss",
        hard_delete_train_prob=0.0,
        include_edit_distance=False,
        regularizer_delay=None,
        target_deletion_rate=None,
        p_controller_value=0.000001
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
        self.loss_function = loss_function
        self.hard_delete_train_prob = hard_delete_train_prob
        self.deletion_threshold = model.config.deletion_threshold
        self.regularizer_delay = regularizer_delay
        self.delete_gate_loss_coeff = args.delete_gate_loss_coeff
        self.target_deletion_rate = target_deletion_rate
        self.p_controller_value = p_controller_value

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
            "layer_attn_scores_pre_gate": [],
            "layer_attn_scores": [],
            "layer_attn_weights": [],
            "layer_value_norms": [],
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
            "eval_layer_attn_scores_pre_gate": [],
            "eval_layer_attn_scores": [],
            "eval_layer_attn_weights": [],
            "eval_layer_value_norms": [],
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

    def p_controller(self, target_deletion, current_deletion, curr_coeff):
        # Calculate error
        error = (target_deletion - current_deletion) / 100
        return max(0.0, curr_coeff + self.p_controller_value * error)

    def __compute_loss(self, outputs, input_ids, log_deletion_metrics=False, log_attn_metrics=False, metrics_prefix=""):

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
            if self.target_deletion_rate is not None and self.state.global_step % 10 == 0:
                self.delete_gate_loss_coeff = self.p_controller(
                    self.target_deletion_rate * 100, percent_non_pad_deleted_tokens, self.delete_gate_loss_coeff)
            loss = cross_entropy_loss + delete_gate_loss

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

        if log_attn_metrics:
            # Get encoder attention logs
            layer = self.model.config.delete_gate_layer+1
            (encoder_attn_scores_pre_gate,
             encoder_attn_scores,
             encoder_attn_weights,
             encoder_value_norms) = outputs.encoder_attn_logs[layer]

            self.metrics[f"{metrics_prefix}layer_attn_scores_pre_gate"].append(
                encoder_attn_scores_pre_gate.mean())
            self.metrics[f"{metrics_prefix}layer_attn_scores"].append(
                encoder_attn_scores.mean())
            self.metrics[f"{metrics_prefix}layer_attn_weights"].append(
                encoder_attn_weights.sum(dim=-1).mean())
            self.metrics[f"{metrics_prefix}layer_value_norms"].append(
                encoder_value_norms.mean())

        return loss, cross_entropy_loss, delete_gate_output

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.pop("input_ids")
        attention_mask = inputs.pop(
            "attention_mask") if "attention_mask" in inputs else None
        labels = inputs.pop("labels")
        if model.training:
            # Log training metrics
            outputs = model(input_ids=input_ids, labels=labels,
                            attention_mask=attention_mask,
                            output_hidden_states=True, output_attn_logs=True,
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
                self.calculate_seq_accuracy(labels, outputs))
            self.metrics["prediction_token_accuracy"].append(
                self.calculate_token_accuracy(labels, outputs))
        else:
            # Log losses for soft deletion
            outputs = model(input_ids=input_ids, labels=labels,
                            attention_mask=attention_mask,
                            output_hidden_states=True, output_attn_logs=True,
                            hard_delete=False)
            loss, cross_entropy_loss, _ = self.__compute_loss(
                outputs, input_ids, log_attn_metrics=True, metrics_prefix="eval_")
            self.metrics["eval_soft_cross_entropy_loss"].append(
                cross_entropy_loss.detach().item())
            self.metrics["eval_soft_total_loss"].append(
                loss.detach().item())

            # Log prediction accuracy
            self.metrics["eval_soft_prediction_seq_accuracy"].append(
                self.calculate_seq_accuracy(labels, outputs))
            self.metrics["eval_soft_prediction_token_accuracy"].append(
                self.calculate_token_accuracy(labels, outputs))

            if self.include_edit_distance:
                self.metrics["eval_soft_edit_distance"].append(
                    self.calculate_edit_distance(labels, outputs))

            # Log losses for hard deletion + other eval metrics
            outputs = model(input_ids=input_ids, labels=labels,
                            attention_mask=attention_mask,
                            output_hidden_states=True, hard_delete=True)
            loss, cross_entropy_loss, _ = self.__compute_loss(
                outputs, input_ids, log_deletion_metrics=True, metrics_prefix="eval_")
            self.metrics[f"eval_hard_cross_entropy_loss"].append(
                cross_entropy_loss.detach().item())
            self.metrics[f"eval_hard_total_loss"].append(
                loss.detach().item())

            # Log prediction accuracy
            self.metrics["eval_hard_prediction_seq_accuracy"].append(
                self.calculate_seq_accuracy(labels, outputs))
            self.metrics["eval_hard_prediction_token_accuracy"].append(
                self.calculate_token_accuracy(labels, outputs))

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
        prediction_seq_accuracy = self.calculate_seq_accuracy(labels, outputs)
        prediction_token_accuracy = self.calculate_token_accuracy(
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
        prediction_seq_accuracy = self.calculate_seq_accuracy(labels, outputs)
        prediction_token_accuracy = self.calculate_token_accuracy(
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

