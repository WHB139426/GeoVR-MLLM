<h2 align="center"> Learning Geometric Representations from Videos for Spatial Intelligent Multimodal Large Language Models </h2>


<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2606.05833-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2606.05833)
[![hf_space](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/WHB139426/GeoVR)

</h5>


<div style="font-family: charter;" align="center">
    <a href="https://whb139426.github.io/" target="_blank">Haibo Wang</a><sup>1</sup>,
    <a href="https://wilburone.github.io/" target="_blank">Lifu Huang</a><sup>1</sup>,
</div>

<div style="font-family: charter;" align="center">
    <sup>1</sup>University of California, Davis&nbsp;&nbsp;
</div>

<div align="center">
  <img src="assets/method.png"/>
</div><br/>

🌟 This is the official repository of the GeoVR, a paradigm to restructure MLLM’s intrinsic representations with geometric awareness using purely 2D videos for Spatial Intelligence. We sharpen our model by incorporating:
- **Multi-Objective Geometric Learning:** Jointly optimizing camera poses, depth maps and metric scales to capture dynamic multi-view consistency and static physical scales.
- **Hierarchical Feature Distillation:** Aligning multi-scale representations from 3D foundation models (e.g., VGGT) to seamlessly bridge low-level geometry and high-level semantics.

## 📝 TODO List

- [x] Release training/evaluation scripts and GeoVR-2B weights
- [ ] Release GeoVR-4B weights
- [ ] Release GeoVR-8B weights
- [ ] Release models trained on datasets mixed with general video understanding tasks

## 🛠️ Install
1. Clone this repository and navigate to folder
```bash
git clone git@github.com:WHB139426/GeoVR-MLLM.git
cd GeoVR-MLLM
```

2. Install Package
```Shell
conda create -n geovr python=3.10.14
conda activate geovr
pip install -r requirements.txt
pip install numpy==1.26.4
pip install flash-attn==2.7.3 --no-build-isolation
```

## 🤗 Prepare the pretrained weights
Set your own `weight_path` to storage the pretrained weights. 

1. Download our released model weights

| Model | Base MLLM | 3D Teacher | Data | VSI-Bench | Download |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `GeoVR-VGGT-Qwen3-VL-2B` | [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | [VGGT-1B](https://huggingface.co/facebook/VGGT-1B) | VSI-590K + VLM-3R-200K | 69.1 | [🤗link](https://huggingface.co/WHB139426/GeoVR/tree/main/GeoVR-VGGT-Qwen3-VL-2B) |
| `GeoVR-Omega-Qwen3-VL-2B` | [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | [VGGT-Omega](https://huggingface.co/facebook/VGGT-Omega) | VSI-590K + VLM-3R-200K | 68.1 | [🤗link](https://huggingface.co/WHB139426/GeoVR/tree/main/GeoVR-Omega-Qwen3-VL-2B) |
| `GeoVR-VGGT-Qwen3-VL-4B` | [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | [VGGT-1B](https://huggingface.co/facebook/VGGT-1B) | VSI-590K + VLM-3R-200K | - | TBD |
| `GeoVR-VGGT-Qwen3-VL-8B` | [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | [VGGT-1B](https://huggingface.co/facebook/VGGT-1B) | VSI-590K + VLM-3R-200K | - | TBD |

2. Download the pretrained weights (Optional, only for training) [[🤗VGGT-Omega](https://huggingface.co/facebook/VGGT-Omega)], [[🤗VGGT-1B](https://huggingface.co/facebook/VGGT-1B)], [[🤗DA3-GIANT-1.1](https://huggingface.co/depth-anything/DA3-GIANT-1.1)], [[🤗DA3METRIC-LARGE](https://huggingface.co/depth-anything/DA3METRIC-LARGE)], [[🤗Qwen3-VL](https://huggingface.co/collections/Qwen/qwen3-vl)] in your own `weight_path`. 

The folder should be organized as follows: 
```
├── GeoVR-MLLM
│   └── models
│   └── training
│   └── utils
│   └── scripts
│   └── vggt
│   └── vggt_omega
│   └── depth_anything_3
│   └── ...
├── weight_path
│   └── GeoVR-VGGT-Qwen3-VL-2B
│   └── VGGT-Omega (Optional, only for training)
│   └── VGGT-1B (Optional, only for training)
│   └── DA3-GIANT-1.1 (Optional, only for training)
│   └── DA3METRIC-LARGE (Optional, only for training)
│   └──...
```


## 🚀 Qucik Start
We give a brief example to run the model with a few lines of code:
```python
import torch
from utils.utils import *
from transformers import AutoProcessor
from models.qwen3vl_geo import Qwen3VLForConditionalGeneration

device = 'cuda:0'
model_id = "/your/path/to/GeoVR-VGGT-Qwen3-VL-2B"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id,
    geometry_encoder_path=None,
    metric_model_path=None,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    add_camera=False,
    add_scale=False,
    add_depth=False,
    distill_geometry_feature=False,
)
model.load_geometric_weights(model_id)
model.to(device)

num_frames = 32
processor = AutoProcessor.from_pretrained(model_id)
processor.video_processor.size = {"longest_edge": 384*num_frames*32*32, "shortest_edge": 4*num_frames*32*32}

messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": './assets/scene0111_02.mp4',},
            {"type": "text", "text": "Measuring distance from the nearest points, select the closest object (trash bin, door, table, refrigerator) to the tv. If multiple exist, use the nearest instance.\nOptions:\nA. trash bin\nB. door\nC. table\nD. refrigerator\nAnswer with the option's letter from the given choices directly."},
        ],
    }
]

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
).to(model.device)

with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **generation_kwargs) 
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
print(output_text)
```


## 📊 Evaluation
1. Download the benchmark [[🤗VSI-Bench](https://huggingface.co/datasets/nyu-visionx/VSI-Bench)] into your own `data_path` and unzip the downloaded files. The folder should be organized as follows: 
```
├── data_path
│   └── VSI-Bench
│       └── test.jsonl
│       └── scannet/
│       └── scannetpp/
│       └── arkitscenes/
│       └──...
```
2. In the script (`scripts/eval.sh`), change the 
    - `MODEL_ID` to `weight_path/GeoVR-VGGT-Qwen3-VL-2B`,
    - `DATA_DIR` to `data_path/VSI-Bench`.
3. Execute the evaluation script. You can easily control the number of GPUs used for parallel inference by modifying `NUM_GPUS` and `CUDA_VISIBLE_DEVICES` within the script.
```bash
bash scripts/eval.sh
```

## 💡 Training
1. Download the training data [[🤗VSI-590K](https://huggingface.co/datasets/nyu-visionx/VSI-590K)], [[🤗VLM-3R-DATA](https://huggingface.co/datasets/Journey9ni/VLM-3R-DATA)] into your own `data_path` and unzip the downloaded files. The folder should be organized as follows:
```
├── data_path
│   └── VSI-590K
│       └── vsi_590k.jsonl
│       └── arkitscenes/
│       └── scannet/
│       └── scannetppv2/
│   └── VLM-3R-DATA
│       └── vlm3r_vsi_205k.json
│       └── vsibench_train/
│       └── vstibench_train/
```
2. In the script (`scripts/train.sh`), change the 
    - `llm` to `weight_path/Qwen3-VL-2B-Instruct`,
    - `geo` to `weight_path/VGGT-1B`, we also support training with other 3D models like VGGT-Omega or DA-3,
    - `metric` to `weight_path/DA3METRIC-LARGE`, 
    - `data_path` to `data_path` ,
    - `output_dir` to `weight_path/checkpoints`.
3. Execute the training script.
```bash
bash scripts/train.sh
```

## ✏️ Citation
If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil:.
```BibTeX
@article{wang2026learning,
  title={Learning Geometric Representations from Videos for Spatial Intelligent Multimodal Large Language Models},
  author={Wang, Haibo and Huang, Lifu},
  journal={arXiv preprint arXiv:2606.05833},
  year={2026}
}
```

## 🤝 Acknowledgement
We are grateful for the following awesome projects our work arising from: [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL), [VGGT-Omega](https://github.com/facebookresearch/vggt-omega), [VGGT](https://github.com/facebookresearch/vggt), [Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3).