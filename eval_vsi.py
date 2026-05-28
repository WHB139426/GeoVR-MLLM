import os
import random
import math
import torch
import json
import torch.multiprocessing as mp
from transformers import AutoProcessor
from tqdm import tqdm
from utils.utils import *
from models.qwen3vl_geo import Qwen3VLForConditionalGeneration
from utils.preprocess import recover_qwen_video_to_numpy, preprocess_geo_frames

def process_chunk(gpu_id, data_chunk, output_file):
    device = f'cuda:{gpu_id}'
    num_frames = int(os.environ['NUM_FRAME'])
    MODEL_ID = os.environ['MODEL_ID']
    GEO_ID = os.environ.get('GEO_ID', None)
    METRIC_ID = os.environ.get('METRIC_ID', None)
    CAMERA = os.environ.get('CAMERA', 'False').lower() == 'true'
    DEPTH = os.environ.get('DEPTH', 'False').lower() == 'true'
    DISTILL = os.environ.get('DISTILL', 'False').lower() == 'true'
    SCALE = os.environ.get('SCALE', 'False').lower() == 'true'
    DATA_DIR=os.environ['DATA_DIR']

    if 'Qwen3-VL' in MODEL_ID:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            geometry_encoder_path=GEO_ID,
            metric_model_path=METRIC_ID,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            add_camera=CAMERA,
            add_depth=DEPTH,
            add_scale=SCALE,
            distill_geometry_feature=DISTILL,
        )
        model.load_geometric_weights(MODEL_ID)
        model.to(device)
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        processor.video_processor.size = {"longest_edge": 384*num_frames*32*32, "shortest_edge": 4*num_frames*32*32}
    else:
        raise ValueError(f'No support {MODEL_ID}')

    result = []    
    for item in tqdm(data_chunk, position=gpu_id, desc=f"GPU {gpu_id}"):
        video_path = os.path.join(DATA_DIR, f"{item['dataset']}/{item['scene_name']}.mp4")
        if item['options'] == None:
            prompt = item['question'] + '\nPlease answer the question ONLY using a single Arabic numeral.'
        else:
            prompt = item['question'] + '\n' + '\n'.join(item['options']) + "\nAnswer with ONLY the option's letter directly."

        if 'qwen' in MODEL_ID.lower():
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": video_path,},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Generate
            generation_kwargs = {
                'do_sample': True,
                'top_p': 0.8,
                'top_k': 20,
                'temperature': 0.7,
                'repetition_penalty': 1.0,
                'max_new_tokens': 32*1024,
            }

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            num_frames=num_frames,
            fps=None,
            enable_thinking=False,
        )
        video = recover_qwen_video_to_numpy(inputs['pixel_values_videos'], inputs['video_grid_thw'], processor)
        pixel_values_geo = preprocess_geo_frames(video)
        inputs['pixel_values_geo'] = pixel_values_geo.unsqueeze(0)
        inputs.to(model.device)

        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            with torch.inference_mode():
                generated_ids = model.generate(**inputs, **generation_kwargs) 
                output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                output_text = output_text.strip()

        result.append({
            'id': item['id'],
            'dataset': item['dataset'],
            'scene_name': item['scene_name'],
            'question_type': item['question_type'],
            'prompt': prompt,
            'pred': output_text,
            'ground_truth': item['ground_truth'],
        })

        save_json(result, output_file)

def main():
    init_seeds(42)
    mp.set_start_method('spawn', force=True)
    num_gpus = int(os.environ['NUM_GPUS'])
    DATA_DIR=os.environ['DATA_DIR']
    MODEL_ID = os.environ['MODEL_ID']

    data = load_jsonl(os.path.join(DATA_DIR, 'test.jsonl'))
    random.shuffle(data)
    chunk_size = math.ceil(len(data) / num_gpus)
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    processes = []
    temp_files = []
    
    print(f"Starting inference on {num_gpus} GPUs...")
    for i in range(num_gpus):
        if i >= len(chunks):
            break
            
        temp_file = f'{MODEL_ID.split("/")[-1]}_result_gpu_{i}.json'
        temp_files.append(temp_file)
        
        p = mp.Process(target=process_chunk, args=(i, chunks[i], temp_file))
        processes.append(p)
        p.start()
        
    for p in processes:
        p.join()
        
    print("All processes completed. Merging results...")
    final_results = []
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            with open(temp_file, 'r', encoding='utf-8') as f:
                gpu_result = json.load(f)
                final_results.extend(gpu_result)
            os.remove(temp_file)
            
    save_json(final_results, 'result.json')
    print(f"Merge successful! Total processed items: {len(final_results)}. Output saved to result.json")

if __name__ == '__main__':
    main()