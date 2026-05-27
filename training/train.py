# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import sys
from pathlib import Path
from transformers import AutoProcessor
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple, Any
from collections.abc import Sequence

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

from training.trainer import Trainer
from training.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from models.qwen3vl_geo import Qwen3VLForConditionalGeneration
from utils.preprocess import IGNORE_INDEX
from utils.utils import *

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.model.language_model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.model.language_model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False
    
    # geometry_encoder always freeze
    for n, p in model.geometry_encoder.named_parameters():
        p.requires_grad = False

    # metric_encoder always freeze
    if model_args.add_scale:
        for n, p in model.metric_encoder.named_parameters():
            p.requires_grad = False


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids, mm_token_type_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids", "mm_token_type_ids")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        mm_token_type_ids = [ids.squeeze(0) for ids in mm_token_type_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        mm_token_type_ids = torch.nn.utils.rnn.pad_sequence(
            mm_token_type_ids, batch_first=True, padding_value=0
        )
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        mm_token_type_ids = mm_token_type_ids[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, :, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            mm_token_type_ids=mm_token_type_ids,
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        frames_geo = list(
            instance["pixel_values_geo"]
            for instance in instances
            if "pixel_values_geo" in instance
        )

        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        if len(frames_geo) != 0:
            batch["pixel_values_geo"] = frames_geo

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = None
        return batch

def make_supervised_data_module(processor, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = []
    
    if 'vsi590k' in data_args.dataset_use:
        from dataset.vsi_590k import VSI_590K
        vsi_590k_dataset = VSI_590K(
            processor=processor, 
            data_args=data_args,
            anno_path = os.path.join(data_args.data_path, 'VSI-590K/vsi_590k.jsonl'),
            video_path = os.path.join(data_args.data_path, 'VSI-590K')
            )
        print('VSI_590K: ', len(vsi_590k_dataset))
        train_dataset.append(vsi_590k_dataset)

    if 'vlm3r' in data_args.dataset_use:
        from dataset.vlm3r import VLM3R
        vlm3r_dataset = VLM3R(
            processor=processor, 
            data_args=data_args,
            anno_vsi_path = os.path.join(data_args.data_path, 'VLM-3R-DATA/vlm3r_vsi_205k.json'),
            video_path = os.path.join(data_args.data_path, 'VSI-590K')
            )
        print('VLM_3R: ', len(vlm3r_dataset))
        train_dataset.append(vlm3r_dataset)

    if len(train_dataset) == 1:
        train_dataset = train_dataset[0]
    else:
        from torch.utils.data import ConcatDataset
        train_dataset = ConcatDataset(train_dataset)
    print('ALL: ', len(train_dataset))

    data_collator = DataCollatorForSupervisedDataset(processor.tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    if "qwen3-vl" in model_args.model_name_or_path.lower():
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            geometry_encoder_path=model_args.geometry_encoder_path,
            metric_model_path=model_args.metric_model_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
            distill_geometry_feature=model_args.distill_geometry_feature,
            add_camera=model_args.add_camera,
            add_depth=model_args.add_depth,
            add_scale=model_args.add_scale,
        )
        model.load_geometric_weights(model_args.model_name_or_path)
        data_args.model_type = "qwen3vl"
        if 'VGGT-1B' in model_args.geometry_encoder_path:
            data_args.geo_type = "vggt"
        elif 'VGGT-Omega' in model_args.geometry_encoder_path:
            data_args.geo_type = "vggt_omega"
        elif 'DA3' in model_args.geometry_encoder_path:
            data_args.geo_type = "da3"
    else:
        raise ValueError(f"No {model_args.model_name_or_path}")


    print(f'the initlized model is {model_args.model_name_or_path} the class is {model.__class__.__name__}')
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path,)
    from utils.video_processing_qwen3_vl import Qwen3VLVideoProcessor
    video_processor = Qwen3VLVideoProcessor.from_pretrained(model_args.model_name_or_path,)
    processor.video_processor = video_processor

    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType
        print("LoRA enabled")

        for p in model.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    else:
        set_model(model_args, model)

        if torch.distributed.get_rank() == 0:
            model.model.visual.print_trainable_parameters()
            model.model.print_trainable_parameters()
            model.print_trainable_parameters()

    
    data_module = make_supervised_data_module(processor, data_args=data_args)
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    init_seeds(42)
    train(attn_implementation="flash_attention_2")