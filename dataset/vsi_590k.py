import os
import sys
import traceback
from torch.utils.data import Dataset
import random
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from utils.utils import *
from utils.preprocess import process_qwen_message_for_sft, update_processor_pixels, recover_qwen_video_to_numpy, get_repeat_video_inputs, preprocess_geo_frames

class VSI_590K(Dataset):
    def __init__(
        self,
        data_args,
        processor,
        anno_path = '/data4/haibo/data/VSI-590K/vsi_590k.jsonl',
        video_path = '/data4/haibo/data/VSI-590K',
        video_only=True
    ):
        super().__init__()
        raw_annos = load_jsonl(anno_path)
        self.video_path = video_path
        self.processor = processor
        self.data_args = data_args
        self.processor = update_processor_pixels(self.processor, self.data_args)
        self.annos = raw_annos

        if video_only:
            self.annos = []
            for item in raw_annos:
                if 'video' in item.keys():
                    self.annos.append(item)

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, i):
        """Handle exceptions since this dataset includes some bad files."""
        try:
            return self.get_item(i)
        except Exception as e:
            traceback.print_exc()
            backup_idx = random.randint(0, len(self) - 1)
            print(self.annos[i])
            print(e)
            print(f"Encounted error when process {i}-th example, use {backup_idx}-th example instead!!!")
            return self.__getitem__(backup_idx)

    def get_item(self, i):
        item = self.annos[i]
        prompt = item['conversations'][0]['value'].replace("<image>\n", "").replace("These are frames of a video.\n", "")
        response = item['conversations'][1]['value']
        messages = []

        if 'video' in item.keys():
            video_path = os.path.join(self.video_path, item['video'])
        elif 'image' in item.keys():
            img_np = np.array(Image.open(os.path.join(self.video_path, item['image'])))
            video_path = np.stack([img_np, img_np], axis=0)

        messages.append({
            "role": "user", 
            "content": 
            [
                {"type": "video", "video": video_path,},
                {"type": "text", "text": prompt},
            ],
        })

        messages.append({
            "role": "assistant",
            "content":
            [
                {"type": "text", "text": response},
            ],
        })

        data_dict = process_qwen_message_for_sft(messages, self.processor, self.data_args)
        video = recover_qwen_video_to_numpy(data_dict['pixel_values_videos'], data_dict['video_grid_thw'], self.processor) 
        if 'video' in item.keys():
            data_dict = get_repeat_video_inputs(data_dict, messages, video, self.processor, self.data_args) # video expand x2  
        pixel_values_geo = preprocess_geo_frames(video, patch_size={'vggt': 14*2, 'da3': 14*2, 'vggt_omega': 16*2}[self.data_args.geo_type], target_size={'vggt': 504, 'da3': 504, 'vggt_omega': max(video.shape[1], video.shape[2])}[self.data_args.geo_type])
        if 'image' in item.keys():
            pixel_values_geo = pixel_values_geo[0].unsqueeze(0)
        data_dict['pixel_values_geo'] = pixel_values_geo
            
        return data_dict

# from dataclasses import dataclass, field
# from typing import Dict, Optional, Sequence, List, Tuple, Any    
# @dataclass
# class DataArguments:
#     max_pixels: int = field(default=32 * 32 * 1024)
#     min_pixels: int = field(default=32 * 32 * 16)
#     video_max_frames: Optional[int] = field(default=32)
#     video_min_frames: Optional[int] = field(default=4)
#     video_max_pixels: int = field(default=32*200*32*32)
#     video_min_pixels: int = field(default=32*16*32*32)
#     video_fps: float = 10000
#     model_type: str = 'qwen3vl'
#     geo_type: str = 'vggt'

# import transformers
# from transformers import AutoProcessor
# parser = transformers.HfArgumentParser(
#     (DataArguments)
# )
# data_args = parser.parse_args_into_dataclasses()[0]

# processor = AutoProcessor.from_pretrained('/data4/haibo/weights/Qwen3-VL-8B-Instruct')
# from utils.video_processing_qwen3_vl import Qwen3VLVideoProcessor
# video_processor = Qwen3VLVideoProcessor.from_pretrained('/data4/haibo/weights/Qwen3-VL-8B-Instruct',)
# processor.video_processor = video_processor

# dataset = VSI_590K(
#     data_args = data_args,
#     processor = processor
# )
# for i in range(10):
#     item = random.choice(dataset)
#     print(dataset.processor.batch_decode(item.input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
#     print(
#           "data_dict['pixel_values_videos'].shape: ", item['pixel_values_videos'].shape,  '\n',
#           "data_dict['video_grid_thw']: ", item['video_grid_thw'],  '\n',
#           "data_dict['pixel_values_geo'].shape: ", item['pixel_values_geo'].shape, '\n',
#           "data_dict['input_ids'].shape: ", item['input_ids'].shape, '\n',
#           "data_dict['labels'].shape: ", item['labels'].shape,  '\n',
#           "data_dict['attention_mask']: ", item['attention_mask'],  '\n',
#           "data_dict['position_ids'].shape: ", item['position_ids'].shape, '\n',
#         )
#     print()
# print(len(dataset))