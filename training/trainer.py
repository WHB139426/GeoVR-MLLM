from typing import Dict, List, Optional, Sequence, Tuple, Callable

import torch
import os
import sys
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers import Trainer
from transformers.cache_utils import Cache
from transformers.utils.deprecation import deprecate_kwarg
from transformers.processing_utils import Unpack
from transformers.utils import logging

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from models.qwen3vl_geo import Qwen3VLVisionModel, Qwen3VLModel, Qwen3VLForConditionalGeneration

logger = logging.get_logger(__name__)

def print_trainable_parameters_visual(self) -> None:
    """
    Prints the trainable status of all vision components including attention blocks and merger module.
    Outputs the indices of trainable/non-trainable blocks and the merger module status.
    """
    trainable_blocks = []
    non_trainable_blocks = []

    # Check trainable status of vision attention blocks
    for block_idx, block in enumerate(self.blocks):
        is_trainable = all(param.requires_grad for param in block.parameters())
        if is_trainable:
            trainable_blocks.append(block_idx)
        else:
            non_trainable_blocks.append(block_idx)

    # Check trainable status of merger module
    is_merger_trainable = any(param.requires_grad for param in self.merger.parameters())

    # Print results
    print("Vision Module - Attention Blocks:")
    print(
        f"Trainable Block Indices: {trainable_blocks if trainable_blocks else 'None'}"
    )
    print(
        f"Non-Trainable Block Indices: {non_trainable_blocks if non_trainable_blocks else 'None'}"
    )
    print(f"Merger Module Trainable: {is_merger_trainable}")

def print_trainable_parameters(self) -> None:
    """
    Prints the trainable status of all LLM components including embeddings, layers, and normalization.
    Outputs the indices of trainable/non-trainable layers and other module statuses.
    """
    # Check embed_tokens
    is_embed_trainable = any(
        param.requires_grad for param in self.language_model.embed_tokens.parameters()
    )
    print(f"LLM Module - Embed Tokens Trainable: {is_embed_trainable}")

    # Check each decoder layer
    trainable_layers = []
    non_trainable_layers = []

    for layer_idx, layer in enumerate(self.language_model.layers):
        is_trainable = any(param.requires_grad for param in layer.parameters())
        if is_trainable:
            trainable_layers.append(layer_idx)
        else:
            non_trainable_layers.append(layer_idx)

    # Print layer status
    print(
        f"LLM Module - Trainable Layer Indices: {trainable_layers if trainable_layers else 'None'}"
    )
    print(
        f"LLM Module - Non-Trainable Layer Indices: {non_trainable_layers if non_trainable_layers else 'None'}"
    )

def print_trainable_parameters_summary(self) -> None:
    """
    Prints the total number of parameters and trainable parameters,
    fully compatible with DeepSpeed ZeRO-3.
    """
    trainable_params = 0
    total_params = 0

    for param in self.parameters():
        num_params = param.ds_numel if hasattr(param, "ds_numel") else param.numel()
        
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params

    trainable_percentage = 100 * trainable_params / total_params if total_params > 0 else 0

    # Print the summary with comma separators for readability
    print(
        f"Model Summary - "
        f"Total params: {total_params:,} || "
        f"Trainable params: {trainable_params:,} || "
        f"Trainable: {trainable_percentage:.4f}%"
    )

def create_optimizer(self):
    opt_model = self.model
    if self.optimizer is None:
        decay_parameters = self.get_decay_parameter_names(opt_model)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        # 1. Map component names to their parameter strings and target learning rates
        param_groups = {
            "camera": (
                [n for n, _ in opt_model.named_parameters() if "camera_head" in n or "camera_token" in n],
                self.args.geo_lr
            ),
            "scale": (
                [n for n, _ in opt_model.named_parameters() if "scale_head" in n or "scale_token" in n],
                self.args.geo_lr
            ),
            "dpt_head": (
                [n for n, _ in opt_model.named_parameters() if "dpt_head" in n],
                self.args.geo_lr
            ),
            "distill_head": (
                [n for n, _ in opt_model.named_parameters() if "distill_head_list" in n],
                self.args.geo_lr
            ),
            "projector": (
                [n for n, _ in opt_model.named_parameters() if "merger" in n],
                self.args.mm_projector_lr
            ),
            "vision_tower": (
                [n for n, _ in opt_model.named_parameters() if "visual" in n],
                self.args.vision_tower_lr
            ),
        }

        optimizer_grouped_parameters = []
        assigned_params = set()

        # 2. Iterate through custom groups and apply specific LRs
        for group_name, (param_names, lr) in param_groups.items():
            if lr is not None and lr != 0:
                optimizer_grouped_parameters.extend([
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() 
                            if n in param_names and n in decay_parameters and p.requires_grad
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() 
                            if n in param_names and n not in decay_parameters and p.requires_grad
                        ],
                        "weight_decay": 0.0,
                        "lr": lr,
                    }
                ])
                assigned_params.update(param_names)

        # 3. Add all remaining base model parameters (everything else)
        optimizer_grouped_parameters.extend([
            {
                "params": [
                    p for n, p in opt_model.named_parameters() 
                    if n not in assigned_params and n in decay_parameters and p.requires_grad
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters() 
                    if n not in assigned_params and n not in decay_parameters and p.requires_grad
                ],
                "weight_decay": 0.0,
            }
        ])

        # Filter out empty parameter groups to prevent optimizer initialization errors
        optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if len(g['params']) > 0]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    return self.optimizer

import torch.nn as nn
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Union
def compute_loss(
    self,
    model: nn.Module,
    inputs: dict[str, torch.Tensor | Any],
    return_outputs: bool = False,
    num_items_in_batch: torch.Tensor | int | None = None,
) -> torch.Tensor | tuple[torch.Tensor, Any]:

    if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
        labels = inputs.pop("labels")
    else:
        labels = None
    if self.model_accepts_loss_kwargs:
        kwargs = {}
        if num_items_in_batch is not None:
            kwargs["num_items_in_batch"] = num_items_in_batch
        inputs = {**inputs, **kwargs}
    outputs = model(**inputs)

    # User-defined compute_loss function
    if self.compute_loss_func is not None:
        if labels is None:
            logger.warning(
                "Trainer: `compute_loss_func` is defined but `labels=None`. "
                "Your custom loss function will still be called with labels=None. "
            )
        loss = self.compute_loss_func(
            outputs,
            labels,
            num_items_in_batch=num_items_in_batch,
        )
    # Default HF loss handling (label smoothing) if no custom loss function
    elif labels is not None:
        unwrapped_model = self.accelerator.unwrap_model(model)
        model_name = (
            unwrapped_model.base_model.model._get_name()
            if _is_peft_model(unwrapped_model)
            else unwrapped_model._get_name()
        )
        if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
            loss = self.label_smoother(outputs, labels, shift_labels=True)
        else:
            loss = self.label_smoother(outputs, labels)
    else:
        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        lm_loss = self.args.lm_loss_weight * outputs["loss"]
        camera_loss = self.args.camera_loss_weight * outputs["camera_loss"]
        depth_loss = self.args.depth_loss_weight * outputs["depth_loss"]
        distill_loss = self.args.distill_loss_weight * outputs["distill_loss"]
        scale_loss = self.args.scale_loss_weight * outputs["scale_loss"]
        loss = lm_loss + camera_loss + depth_loss + distill_loss + scale_loss

    if (
        self.args.average_tokens_across_devices
        and (self.model_accepts_loss_kwargs or self.compute_loss_func)
        and num_items_in_batch is not None
    ):
        loss *= self.accelerator.num_processes if self.args.n_gpu <= 1 else self.args.n_gpu

    # log loss
    self._metrics["lm_loss"].append(self.accelerator.gather_for_metrics(lm_loss.detach()).mean().item())
    self._metrics["camera_loss"].append(self.accelerator.gather_for_metrics(camera_loss.detach()).mean().item())
    self._metrics["depth_loss"].append(self.accelerator.gather_for_metrics(depth_loss.detach()).mean().item())
    self._metrics["distill_loss"].append(self.accelerator.gather_for_metrics(distill_loss.detach()).mean().item())
    self._metrics["scale_loss"].append(self.accelerator.gather_for_metrics(scale_loss.detach()).mean().item())
    self._metrics["depth_loss_reg"].append(self.accelerator.gather_for_metrics(self.args.depth_loss_weight * outputs["depth_loss_reg"].detach()).mean().item())
    self._metrics["depth_loss_grad"].append(self.accelerator.gather_for_metrics(self.args.depth_loss_weight * outputs["depth_loss_grad"].detach()).mean().item())
    self._metrics["camera_loss_T"].append(self.accelerator.gather_for_metrics(self.args.camera_loss_weight * outputs["camera_loss_T"].detach()).mean().item())
    self._metrics["camera_loss_R"].append(self.accelerator.gather_for_metrics(self.args.camera_loss_weight * outputs["camera_loss_R"].detach()).mean().item())
    self._metrics["camera_loss_FL"].append(self.accelerator.gather_for_metrics(self.args.camera_loss_weight * outputs["camera_loss_FL"].detach()).mean().item())

    return (loss, outputs) if return_outputs else loss

def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
    metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
    logs = {**logs, **metrics}
    if self.state.epoch is not None:
        logs["epoch"] = self.state.epoch

    if len(self.optimizer.param_groups) > 1:
            base_lr = self.optimizer.param_groups[-1]["lr"]
            geo_lr = self.optimizer.param_groups[0]["lr"]
            logs["base_lr"] = base_lr
            logs["geo_lr"] = geo_lr            
            logs["learning_rate"] = base_lr

    output = {**logs, **{"step": self.state.global_step}}
    self.state.log_history.append(output)
    self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    # === IMPORTANT: Clear the metrics buffer after logging ===
    self._metrics.clear()

# # ==========================================
# # rewrite Trainer Dataloader Sampler logic: fix deepspeed zero3 hang on by avoid mixing image and video within one micro-batch
# # refer to: https://github.com/QwenLM/Qwen3-VL/issues/126
# # ==========================================
# from train.sampler import ModalityAwareSampler
# from torch.utils.data import Dataset
# def _get_train_sampler(self, train_dataset: Dataset | None = None) -> torch.utils.data.Sampler | None:
#     if train_dataset is None:
#         train_dataset = self.train_dataset
#     return ModalityAwareSampler(
#         data_source=train_dataset,
#         batch_size=self.args.train_batch_size,
#     )
# Trainer._get_train_sampler = _get_train_sampler

# Apply monkey patches

from collections import defaultdict
Trainer._metrics = defaultdict(list)
Trainer.compute_loss = compute_loss
Trainer.log = log

Trainer.create_optimizer = create_optimizer
Qwen3VLVisionModel.print_trainable_parameters = (print_trainable_parameters_visual)
Qwen3VLModel.print_trainable_parameters = print_trainable_parameters
Qwen3VLForConditionalGeneration.print_trainable_parameters = print_trainable_parameters_summary