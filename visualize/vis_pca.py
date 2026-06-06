import os
import sys
import random
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import AutoProcessor

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

# Custom imports
from models.qwen3vl_geo import Qwen3VLForConditionalGeneration
from utils.preprocess import recover_qwen_video_to_numpy, preprocess_geo_frames

# ==========================================
# CONFIGURATION
# ==========================================
DEVICE = 'cuda:0'
QWEN_WEIGHTS = '/your/path/to/GeoVR-Omega-Qwen3-VL-2B'
VGGT_WEIGHTS = '/your/path/to/VGGT-Omega/vggt_omega_1b_512.pt'
QWEN_WEIGHTS_ORIGINAL = '/your/path/to/Qwen3-VL-2B-Instruct'

NUM_FRAMES = 8
DEPTH_RATIO = 0.5
vggt_patch_size = 16 if 'Omega' in VGGT_WEIGHTS else 14

# ==========================================
# VISUALIZATION FUNCTIONS
# ==========================================

def visualize_video_local_pca_comparison(pixel_values_vggt, pixel_values_qwen, hidden_states_vggt, hidden_states_qwen, hidden_states_vanilla, patch_size_vggt=16, patch_size_qwen=32):
    T = min(pixel_values_vggt.shape[1], pixel_values_qwen.shape[1], hidden_states_vggt.shape[1], hidden_states_qwen.shape[1], hidden_states_vanilla.shape[1])
    fig, axes = plt.subplots(T, 4, figsize=(16, 3 * T))
    if T == 1: 
        axes = np.expand_dims(axes, axis=0)
    for i in range(T):
        img_qwen = pixel_values_qwen[0, i].permute(1, 2, 0).cpu().numpy()
        img_qwen = (img_qwen * 255).astype(np.uint8) if img_qwen.max() <= 1.0 else img_qwen.astype(np.uint8)
        H_q, W_q, _ = img_qwen.shape
        
        axes[i, 0].imshow(img_qwen)
        axes[i, 0].set_title("Original" if i == 0 else "")
        axes[i, 0].set_ylabel(f"Frame {i}", fontsize=14, fontweight='bold')
        axes[i, 0].set_xticks([]) 
        axes[i, 0].set_yticks([])
    
        H_v, W_v = pixel_values_vggt.shape[3], pixel_values_vggt.shape[4]
        pca_vggt_frame = PCA(n_components=3).fit_transform(hidden_states_vggt[0, i].float().cpu().numpy())
        pca_vggt_norm = (pca_vggt_frame - pca_vggt_frame.min(axis=0)) / (pca_vggt_frame.max(axis=0) - pca_vggt_frame.min(axis=0) + 1e-8)
        pca_map_vggt_resized = cv2.resize(pca_vggt_norm.reshape(H_v // patch_size_vggt, W_v // patch_size_vggt, 3), (W_v, H_v), interpolation=cv2.INTER_LINEAR)
        
        axes[i, 1].imshow(pca_map_vggt_resized)
        axes[i, 1].set_title("VGGT-Omega Local PCA" if i == 0 else "") 
        axes[i, 1].axis('off')

        pca_qwen_frame = PCA(n_components=3).fit_transform(hidden_states_qwen[0, i].float().cpu().numpy())
        pca_qwen_norm = (pca_qwen_frame - pca_qwen_frame.min(axis=0)) / (pca_qwen_frame.max(axis=0) - pca_qwen_frame.min(axis=0) + 1e-8)
        pca_map_qwen_resized = cv2.resize(pca_qwen_norm.reshape(H_q // patch_size_qwen, W_q // patch_size_qwen, 3), (W_q, H_q), interpolation=cv2.INTER_LINEAR)
        
        axes[i, 2].imshow(pca_map_qwen_resized)
        axes[i, 2].set_title("Fine-tuned Qwen Local PCA" if i == 0 else "") 
        axes[i, 2].axis('off')

        pca_vanilla_frame = PCA(n_components=3).fit_transform(hidden_states_vanilla[0, i].float().cpu().numpy())
        pca_vanilla_norm = (pca_vanilla_frame - pca_vanilla_frame.min(axis=0)) / (pca_vanilla_frame.max(axis=0) - pca_vanilla_frame.min(axis=0) + 1e-8)
        pca_map_vanilla_resized = cv2.resize(pca_vanilla_norm.reshape(H_q // patch_size_qwen, W_q // patch_size_qwen, 3), (W_q, H_q), interpolation=cv2.INTER_LINEAR)
        
        axes[i, 3].imshow(pca_map_vanilla_resized)
        axes[i, 3].set_title("Vanilla Qwen Local PCA" if i == 0 else "") 
        axes[i, 3].axis('off')

    plt.subplots_adjust(left=0.05, right=0.98, bottom=0.02, top=0.95, wspace=0.05, hspace=0.1)

    save_path = 'pca.png'
    plt.savefig(save_path, dpi=450, bbox_inches='tight', pad_inches=0.05)
    plt.show()

def main():
    # 1. Load Models
    print("Loading Qwen model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        QWEN_WEIGHTS, geometry_encoder_path=VGGT_WEIGHTS, dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", distill_geometry_feature=False,
        add_camera=False, add_depth=False, add_scale=False,
    ).to(DEVICE)

    print("Loading VGGT model...")
    vggt = model.geometry_encoder


    vanilla_model = Qwen3VLForConditionalGeneration.from_pretrained(
        QWEN_WEIGHTS_ORIGINAL, geometry_encoder_path=None, dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", distill_geometry_feature=False,
        add_camera=False, add_depth=False, add_scale=False,
    ).to(DEVICE)

    processor = AutoProcessor.from_pretrained(QWEN_WEIGHTS_ORIGINAL)
    processor.video_processor.size = {"longest_edge": 300*2*NUM_FRAMES*32*32, "shortest_edge": 4*2*NUM_FRAMES*32*32}
    processor.video_processor.fps = 10000

    # 2. Load Data
    print("Loading data...")
    video_path = './assets/scene0111_02.mp4'
    prompt = 'N/A.'
    # 3. Process Original Video for VGGT
    messages = [
        {"role": "user", "content": [{"type": "video", "video": video_path}, {"type": "text", "text": prompt}]}
    ]
    inputs_vggt = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, 
        return_tensors="pt", num_frames=NUM_FRAMES, fps=None, enable_thinking=False
    ).to(DEVICE)

    video_np = recover_qwen_video_to_numpy(inputs_vggt['pixel_values_videos'], inputs_vggt['video_grid_thw'], processor)
    target_size = int(max(inputs_vggt['video_grid_thw'][0])*16) if vggt_patch_size == 16 else 504
    pixel_values_vggt = preprocess_geo_frames(video_np, patch_size=vggt_patch_size*2, target_size=target_size).unsqueeze(0).to(DEVICE) 
    print(pixel_values_vggt.shape)
    
    # 4. Save Extracted Frames to Video
    OUTPUT_DIR = 'pca_images'
    FPS = 2
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    num_unique_frames, height, width, _ = video_np.shape
    output_path = f"{OUTPUT_DIR}/repeated_video.mp4"
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (width, height))
    
    for i, frame in enumerate(video_np):
        frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"frame_{i:02d}.png"), frame_bgr)
        out.write(frame_bgr)
        out.write(frame_bgr) # Repeated write based on original script logic
    out.release()
    print(f"✅ Processed {num_unique_frames} frames. Saved to {output_path}")

    # 5. Process Repeated Video for Qwen
    messages[0]["content"][0]["video"] = output_path
    inputs_qwen = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, 
        return_tensors="pt", num_frames=NUM_FRAMES*2, fps=None, enable_thinking=False
    ).to(DEVICE)
    print(inputs_qwen['video_grid_thw'])

    # 6. Extract Hidden States
    print("Extracting features...")
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16), torch.inference_mode():
        # VGGT Features
        aggregated_tokens_list = vggt(pixel_values_vggt)['aggregated_tokens_list']
        target_layer_idx_vggt = int(DEPTH_RATIO * 24) - 1
        patch_start_idx = 5 if 'VGGT-1B' in VGGT_WEIGHTS else 17
        last_hidden_states_vggt = aggregated_tokens_list[target_layer_idx_vggt][:, :, patch_start_idx:, :1024] 

        # Qwen Features
        outputs = model(**inputs_qwen)
        all_layers_video_features = outputs['all_layers_video_features']
        target_layer_idx_qwen = int(DEPTH_RATIO * (len(all_layers_video_features) - 1))
        last_hidden_states_qwen = all_layers_video_features[target_layer_idx_qwen][0].unsqueeze(0)

        # Vanilla Qwen Features
        outputs = vanilla_model(**inputs_qwen)
        all_layers_video_features = outputs['all_layers_video_features']
        target_layer_idx_qwen = int(DEPTH_RATIO * (len(all_layers_video_features) - 1))
        last_hidden_states_vanilla = all_layers_video_features[target_layer_idx_qwen][0].unsqueeze(0)

    pixel_values_qwen = torch.from_numpy(video_np).permute(0, 3, 1, 2).unsqueeze(0)

    # 7. Execute Visualizations
    print("Generating visualizations...")
    
    visualize_video_local_pca_comparison(
        pixel_values_vggt=pixel_values_vggt, pixel_values_qwen=pixel_values_qwen, 
        hidden_states_vggt=last_hidden_states_vggt, hidden_states_qwen=last_hidden_states_qwen, hidden_states_vanilla=last_hidden_states_vanilla,
        patch_size_vggt=vggt_patch_size, patch_size_qwen=32
    )


if __name__ == "__main__":
    main()