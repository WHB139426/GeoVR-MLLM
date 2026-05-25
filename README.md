<h2 align="center"> GeoVR: Learning Geometric Video Representations for Spatial Intelligence within Multimodal Large Language Models </h2>

🌟 This is the official repository of the GeoVR, a paradigm to restructure MLLM’s intrinsic representations with geometric awareness using purely 2D videos for Spatial Intelligence.

<div align="center">
  <img src="assets/method.png"/>
</div><br/>

## 📝 TODO List

- [-] Release model weights and inference code
- [x] Release model weights and inference code

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

2. Download the pretrained weights (Optional, only for training) [[🤗VGGT-Omega](https://huggingface.co/facebook/VGGT-Omega)], [[🤗VGGT-1B](https://huggingface.co/facebook/VGGT-1B)], [[🤗DA3-GIANT-1.1](https://huggingface.co/depth-anything/DA3-GIANT-1.1)], [[🤗DA3METRIC-LARGE](https://huggingface.co/depth-anything/DA3METRIC-LARGE)], [[🤗Qwen3-VL](https://huggingface.co/collections/Qwen/qwen3-vl)] in your own `weight_path`. 

The folder should be organized as follows: 
```
├── GeoVR
│   └── models
│   └── training
│   └── utils
│   └── scripts
│   └── vggt
│   └── vggt_omega
│   └── depth_anything_3
│   └── ...
├── weight_path
│   └── Qwen3-VL-2B-Instruct
│   └── Qwen3-VL-4B-Instruct
│   └── Qwen3-VL-8B-Instruct
│   └── VGGT-Omega (Optional, only for training)
│   └── VGGT-1B (Optional, only for training)
│   └── DA3-GIANT-1.1 (Optional, only for training)
│   └── DA3METRIC-LARGE (Optional, only for training)
│   └──...
```


## 🚀 Qucik Start
We give a brief example to run the example code.

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
2. 

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
│       └── vlm3r_vst_132k.json
│       └── vsibench_train/
│       └── vstibench_train/
```
2. 

## ✏️ Citation
If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil:.
```BibTeX

```

## 🤝 Acknowledgement
We are grateful for the following awesome projects our work arising from: [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL), [VGGT-Omega](https://github.com/facebookresearch/vggt-omega), [VGGT](https://github.com/facebookresearch/vggt), [Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3).