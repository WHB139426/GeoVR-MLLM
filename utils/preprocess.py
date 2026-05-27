import logging
import os
import sys
from typing import Dict, Optional, Sequence, List, Tuple, Any
from collections.abc import Sequence
import torch
import einops
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from utils.rope2d import get_rope_index_3

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def update_processor_pixels(processor, data_args):
    logger = logging.getLogger(__name__)

    # --- Image Processor ---
    ip = processor.image_processor
    rank0_print("=== BEFORE IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"ip.size: {ip.size}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    if hasattr(ip, "min_pixels") and hasattr(ip, "max_pixels"):
        ip.min_pixels = data_args.min_pixels
        ip.max_pixels = data_args.max_pixels
        rank0_print(f"✅ Updated image_processor min_pixels to {data_args.min_pixels}")
        rank0_print(f"✅ Updated image_processor max_pixels to {data_args.max_pixels}")

    if hasattr(ip, "size"):
        ip.size["shortest_edge"] = data_args.min_pixels
        ip.size["longest_edge"] = data_args.max_pixels
        rank0_print(
            f"✅ Updated image_processor size['shortest_edge'] to {data_args.min_pixels}"
        )
        rank0_print(
            f"✅ Updated image_processor size['longest_edge'] to {data_args.max_pixels}"
        )

    rank0_print("=== AFTER IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    # --- Video Processor ---
    if hasattr(processor, "video_processor") and processor.video_processor is not None:
        vp = processor.video_processor
        rank0_print("\n=== BEFORE VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(
            f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
        )
        rank0_print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

        if hasattr(vp, "min_pixels") and hasattr(vp, "max_pixels"):
            vp.min_pixels = data_args.video_min_pixels
            vp.max_pixels = data_args.video_max_pixels
            rank0_print(
                f"✅ Updated video_processor min_pixels to {data_args.video_min_pixels}"
            )
            rank0_print(
                f"✅ Updated video_processor max_pixels to {data_args.video_max_pixels}"
            )

        if hasattr(vp, "min_frames") and hasattr(vp, "max_frames"):
            vp.min_frames = data_args.video_min_frames
            vp.max_frames = data_args.video_max_frames
            rank0_print(
                f"✅ Updated video_processor min_frames to {data_args.video_min_frames}"
            )
            rank0_print(
                f"✅ Updated video_processor max_frames to {data_args.video_max_frames}"
            )

        if hasattr(vp, "fps"):
            vp.fps = data_args.video_fps
            rank0_print(f"✅ Updated video_processor fps to {data_args.video_fps}")

        if hasattr(vp, "size"):
            vp.size["shortest_edge"] = data_args.video_min_pixels
            vp.size["longest_edge"] = data_args.video_max_pixels
            rank0_print(
                f"✅ Updated Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
            )
            rank0_print(
                f"✅ Updated Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}"
            )

        rank0_print("=== AFTER VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(
            f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
        )
        rank0_print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

    return processor

def preprocess_qwen_visual(
    messages,
    processor,
) -> Dict:

    full_result = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt"
    )

    input_ids = full_result["input_ids"]
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids).unsqueeze(0)
        
    # create labels
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    input_ids_flat = input_ids[0].tolist()
    L = len(input_ids_flat)
    pos = 0
    while pos < L:
        if input_ids_flat[pos] == 77091:
            ans_start = pos + 2
            ans_end = ans_start
            while ans_end < L and input_ids_flat[ans_end] != 151645:
                ans_end += 1
            if ans_end < L:
                labels[0, ans_start : ans_end + 2] = input_ids[
                    0, ans_start : ans_end + 2
                ]
                pos = ans_end
        pos += 1
        
    full_result["labels"] = labels
    full_result["input_ids"] = input_ids
    return full_result

def process_qwen_message_for_sft(messages, processor, data_args):

    # # 0. Update processor
    # processor = update_processor_pixels(processor, data_args)

    # 1. processor's chat template
    data_dict = preprocess_qwen_visual(
        messages,
        processor,
    )

    # 2. image_grid_thw
    if "image_grid_thw" in data_dict:
        grid_thw = data_dict.get("image_grid_thw")
        if not isinstance(grid_thw, Sequence):
            grid_thw = [grid_thw]
    else:
        grid_thw = None

    # 3. video_grid_thw && second_per_grid_ts
    if "video_grid_thw" in data_dict:
        video_grid_thw = data_dict.get("video_grid_thw")
        if not isinstance(video_grid_thw, Sequence):
            video_grid_thw = [video_grid_thw]
        second_per_grid_ts = [
            processor.video_processor.temporal_patch_size
            / processor.video_processor.fps
        ] * len(video_grid_thw)
    else:
        video_grid_thw = None
        second_per_grid_ts = None

    # 4. position_ids (RoPE 3D index)
    if data_args.model_type == "qwen3vl" or data_args.model_type == "qwen3_5":
        get_rope_index = get_rope_index_3
    else:
        raise ValueError(f"model_type: {data_args.model_type} not supported")
    merge_size = getattr(processor.image_processor, "merge_size", 2)
    position_ids, _ = get_rope_index(
        merge_size,
        data_dict["input_ids"],
        image_grid_thw=torch.cat(grid_thw, dim=0) if grid_thw else None,
        video_grid_thw=(
            torch.cat(video_grid_thw, dim=0) if video_grid_thw else None
        ),
        second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
    )

    seq_len = data_dict["input_ids"][0].size(0)
    data_dict["position_ids"] = position_ids
    data_dict["attention_mask"] = [seq_len]
    
    return data_dict

def recover_qwen_video_to_numpy(pixel_values_videos: torch.Tensor, 
                                video_grid_thw: torch.Tensor, 
                                processor) -> np.ndarray:
    
    image_processor = getattr(processor, "image_processor", processor)
    mean = torch.tensor(image_processor.image_mean, device=pixel_values_videos.device)
    std = torch.tensor(image_processor.image_std, device=pixel_values_videos.device)
    patch_size = image_processor.patch_size               
    temporal_patch_size = image_processor.temporal_patch_size 
    merge_size = getattr(image_processor, "merge_size", 2)
    
    t_grid, h_grid, w_grid = video_grid_thw.view(-1).tolist()
    video_tensor = einops.rearrange(
        pixel_values_videos,
        "(t h_macro w_macro m_h m_w) (c t_p p_h p_w) -> (t t_p) (h_macro m_h p_h) (w_macro m_w p_w) c",
        t=t_grid, 
        h_macro=h_grid // merge_size, 
        w_macro=w_grid // merge_size, 
        m_h=merge_size, 
        m_w=merge_size,
        c=3, 
        t_p=temporal_patch_size, 
        p_h=patch_size, 
        p_w=patch_size
    )
    
    mean = mean.view(1, 1, 1, 3)
    std = std.view(1, 1, 1, 3)
    
    video_tensor = (video_tensor * std) + mean
    
    video_tensor = (video_tensor * 255.0).clamp(0, 255).to(torch.uint8)
    video_numpy = video_tensor.cpu().numpy()
    
    return video_numpy

import re
def extract_last_second(text: str) -> float:
    matches = re.findall(r'<([0-9.]+) seconds>', text)
    if matches:
        return float(matches[-1])
    else:
        return None
    
import copy
def get_repeat_video_inputs(data_dict, messages, video, processor, data_args):
    old_video = video
    video = np.repeat(video, repeats=2, axis=0)
    video_duration = extract_last_second(processor.batch_decode(data_dict.input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
    if video_duration != None and video_duration != 0:
        sample_fps = video.shape[0]/video_duration
    else:
        sample_fps = 2

    video_path = messages[0]['content'][0]['video']
    messages[0]['content'][0]['video'] = video
    temp_processor = copy.deepcopy(processor)
    temp_processor.video_processor.size["longest_edge"] = processor.video_processor.size["longest_edge"] * 2
    temp_processor.video_processor.max_frames = processor.video_processor.max_frames * 2
    temp_processor.video_processor.fps = 10000
    temp_processor.video_processor.sample_fps = sample_fps
    data_dict = process_qwen_message_for_sft(messages, temp_processor, data_args)
    return data_dict

def preprocess_geo_frames(video_array: np.ndarray, mode="crop", patch_size=14, target_size=518) -> torch.Tensor:
    """
    Preprocess a numpy array of video frames for model input.
    
    Args:
        video_array (np.ndarray): Video frames of shape (T, H, W, 3) in RGB format.
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets longest edge to 518px, preserves aspect ratio. No actual cropping.
                             - "pad": Sets longest edge to 518px and pads the shorter edge with white to make it 518x518.

    Returns:
        torch.Tensor: Batched tensor of preprocessed frames with shape (T, 3, H, W)
    """
    if video_array.shape[0] == 0:
        raise ValueError("Video array must contain at least 1 frame")

    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    target_size = target_size

    # Iterate through each frame in the T dimension
    for frame in video_array:
        # Convert numpy array (H, W, 3) to PIL Image
        # Ensure it's uint8 to avoid PIL conversion errors
        img = Image.fromarray(frame.astype(np.uint8))

        width, height = img.size

        if width >= height:
            new_width = target_size
            new_height = round(height * (new_width / width) / patch_size) * patch_size
        else:
            new_height = target_size
            new_width = round(width * (new_height / height) / patch_size) * patch_size

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = TF.to_tensor(img)  # Convert to tensor (0, 1)


        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    return images