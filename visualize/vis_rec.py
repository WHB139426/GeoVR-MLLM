import os
import sys
import random
import cv2
import torch
import numpy as np
from transformers import AutoProcessor
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

# Custom imports
from models.qwen3vl_geo import Qwen3VLForConditionalGeneration
from utils.preprocess import recover_qwen_video_to_numpy, preprocess_geo_frames

# ==========================================
# CONFIGURATION
# ==========================================
DEVICE = 'cuda:0'
QWEN_WEIGHTS = '/your/path/to/GeoVR-VGGT-Qwen3-VL-2B'
VGGT_WEIGHTS = '/your/path/to/VGGT-1B'
METRIC_WITHTS = '/your/path/to/DA3METRIC-LARGE'
vggt_patch_size = 16 if 'Omega' in VGGT_WEIGHTS else 14


NUM_FRAMES = 32
FPS = 2

print("Loading Qwen model...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    QWEN_WEIGHTS, geometry_encoder_path=VGGT_WEIGHTS, metric_model_path=METRIC_WITHTS, dtype=torch.bfloat16,
    attn_implementation="flash_attention_2", distill_geometry_feature=False,
    add_camera=True, add_depth=True, add_scale=True,
)
model.load_geometric_weights(QWEN_WEIGHTS)
model.to(DEVICE)
vggt = model.geometry_encoder
model.train()

processor = AutoProcessor.from_pretrained(QWEN_WEIGHTS)
processor.video_processor.size = {"longest_edge": 384*2*NUM_FRAMES*32*32, "shortest_edge": 4*2*NUM_FRAMES*32*32}
processor.video_processor.fps = 10000

# 2. Load Data
video_path = './assets/scene0101_02.mp4'
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
pixel_values_geo = preprocess_geo_frames(video_np, patch_size=vggt_patch_size*2, target_size=target_size).unsqueeze(0).to(DEVICE) 

# 4. Save Extracted Frames to Video
OUTPUT_DIR = 'rec_images'
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
inputs_qwen['pixel_values_geo'] = pixel_values_geo


# 6. Extract Hidden States
print("Extracting features...")
with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
    # VGGT
    print(pixel_values_geo.shape)
    _, gt_pose_enc_list, _, gt_depth_list, gt_depth_conf_list = model.get_geo_features(pixel_values_geo=pixel_values_geo, distill_visual_indexes=[23])
    pose_enc = gt_pose_enc_list[0]
    depth = gt_depth_list[0]
    depth_conf = gt_depth_conf_list[0]
    scale_factor = model.get_gt_scale(pixel_values_geo, gt_pose_enc_list, gt_depth_list, gt_depth_conf_list)[0]
    print(pose_enc.shape, depth.shape, depth_conf.shape, scale_factor)
    geo_predictions = {
        "images": pixel_values_geo,
        "pose_enc": pose_enc,
        "depth": depth,
        "depth_conf": depth_conf,
        "scale": scale_factor,
    }

    # Qwen3-VL
    print(inputs_qwen['pixel_values_videos'].shape, inputs_qwen['video_grid_thw'])
    outputs = model(**inputs_qwen)
    pred_pose_enc = outputs['pred_pose_enc_list'][0].float().detach().cpu()
    pred_depth = outputs['pred_depth_list'][0].float().detach().cpu()
    pred_scale = outputs['pred_scale'][0].float().detach().cpu()
    print(pred_pose_enc.shape, pred_depth.shape, pred_scale)
    mllm_predictions = {
        "images": pixel_values_geo,
        "pose_enc": pred_pose_enc,
        "depth": pred_depth,
        "depth_conf": depth_conf, 
        "scale": pred_scale,
    }
    

# 7. Reconstruction
import open3d as o3d
from PIL import Image
print("Reconstruction ...")
def reconstruct_and_save(predictions, model, prefix="vggt"):
    print(f"\n[{prefix.upper()}] Reconstruction ...")
    images = predictions['images']
    B, S, C, H, W = images.shape
    conf_thresh_percentile = 40
    
    if model.config.geo_type == 'vggt':
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions['pose_enc'], (H, W))
    elif model.config.geo_type == 'vggt_omega':
        from vggt_omega.utils.pose_enc import pose_encoding_to_extri_intri
        extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions['pose_enc'], (H, W))
    else:
        raise ValueError(f"Unknown geo_type: {model.config.geo_type}")

    depths = predictions['depth'].clone()
    scale = predictions['scale']
    depths *= scale
    extrinsics_scaled = extrinsics.clone()
    extrinsics_scaled[:, :, :3, 3] *= scale

    base_out_dir = OUTPUT_DIR
    img_dir = os.path.join(base_out_dir, f"{prefix}_images")
    os.makedirs(img_dir, exist_ok=True)

    merged_pcd = o3d.geometry.PointCloud()
    camera_data = []

    for i in range(S):
        img_tensor = images[0, i].permute(1, 2, 0).contiguous().cpu().numpy()
        img_np = (img_tensor * 255).astype(np.uint8)
        color_img = o3d.geometry.Image(img_np)
        
        img_filename = os.path.join(img_dir, f"cam_{i}.jpg")
        Image.fromarray(img_np).save(img_filename)
        
        depth_tensor = depths[0, i, :, :].float().contiguous().cpu().numpy()
        conf_tensor = predictions['depth_conf'][0, i].float().cpu().numpy()
        
        conf_thres_value = np.percentile(conf_tensor, conf_thresh_percentile) 
        depth_tensor[conf_tensor < conf_thres_value] = 0.0
        depth_img = o3d.geometry.Image(depth_tensor)
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img, 
            depth_img, 
            depth_scale=1.0, 
            depth_trunc=15.0,  
            convert_rgb_to_intensity=False
        )
        
        extrinsic_3x4 = extrinsics_scaled[0, i].float().cpu().numpy()
        extrinsic_mat = np.eye(4, dtype=np.float64)
        extrinsic_mat[:3, :] = extrinsic_3x4
        
        intrinsic_mat = intrinsics[0, i].float().cpu().numpy().astype(np.float64)
        o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=W, height=H,
            fx=intrinsic_mat[0, 0], fy=intrinsic_mat[1, 1],
            cx=intrinsic_mat[0, 2], cy=intrinsic_mat[1, 2]
        )
        
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, o3d_intrinsic, extrinsic_mat
        )
        merged_pcd += pcd
        
        camera_data.append({
            "id": i,
            "extrinsic": extrinsic_mat.tolist(),
            "intrinsic": intrinsic_mat.tolist(),
            "W": W,
            "H": H,
            "image_path": f"{prefix}_images/cam_{i}.jpg"
        })

    merged_pcd = merged_pcd.voxel_down_sample(voxel_size=0.01)
    ply_output_path = os.path.join(f"{prefix}_reconstruction.ply")
    o3d.io.write_point_cloud(ply_output_path, merged_pcd)

    camera_output_path = os.path.join(f"{prefix}_cameras.json")
    with open(camera_output_path, "w") as f:
        json.dump(camera_data, f, indent=4)

reconstruct_and_save(mllm_predictions, model, prefix="mllm")
reconstruct_and_save(geo_predictions, model, prefix="geo")