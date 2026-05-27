import os
import sys
import traceback
from torch.utils.data import Dataset
import random
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from utils.utils import *
from utils.preprocess import process_qwen_message_for_sft, update_processor_pixels, recover_qwen_video_to_numpy, get_repeat_video_inputs, preprocess_geo_frames

class VLM3R(Dataset):
    def __init__(
        self,
        data_args,
        processor,
        anno_vsi_path = '/your/path/to/VLM-3R-DATA/vlm3r_vsi_205k.json',
        video_path = '/your/path/to/VSI-590K',
    ):
        super().__init__()
        raw_annos = load_json(anno_vsi_path)
        self.video_path = video_path
        self.processor = processor
        self.data_args = data_args
        self.processor = update_processor_pixels(self.processor, self.data_args)
        self.annos = raw_annos

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
        if item['data_source'] == 'scannetpp':
            item['data_source'] = 'scannetppv2'
        video_path = os.path.join(self.video_path, item['data_source'], item['scene_name']+'.mp4')

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
        data_dict = get_repeat_video_inputs(data_dict, messages, video, self.processor, self.data_args) # video expand x2  
        pixel_values_geo = preprocess_geo_frames(video, patch_size={'vggt': 14*2, 'da3': 14*2, 'vggt_omega': 16*2}[self.data_args.geo_type], target_size={'vggt': 504, 'da3': 504, 'vggt_omega': 512}[self.data_args.geo_type])
        data_dict['pixel_values_geo'] = pixel_values_geo
            
        return data_dict