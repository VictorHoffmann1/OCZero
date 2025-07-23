set -ex
export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export CUDA_VISIBLE_DEVICES=0

python main.py --env BreakoutNoFrameskip-v4 --case atari --opr train --force \
  --num_gpus 1 --num_cpus 16 --cpu_actor 1 --gpu_actor 1 \
  --object_store_memory 4294967296 \
  --seed 0 \
  --p_mcts_num 4 \
  --use_priority \
  --use_max_priority \
  --amp_type 'torch_amp' \
  --info 'EfficientZero-V1'
