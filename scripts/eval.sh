export MODEL_ID='/your/path/to/GeoVR-Qwen3-VL-2B'
export DATA_DIR="/your/path/to/VSI-Bench"

export NUM_GPUS=2

export CAMERA=False
export DEPTH=False
export DISTILL=False
export SCALE=False

export NUM_FRAME=128
CUDA_VISIBLE_DEVICES=0,1 python eval_vsi.py

python metric_result.py