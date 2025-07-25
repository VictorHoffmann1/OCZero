set -ex
export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export CUDA_VISIBLE_DEVICES=0

python main.py --env BreakoutNoFrameskip-v4 --case atari --opr train --force \
  --num_gpus 1 --num_cpus 16 --cpu_actor 3 --gpu_actor 3 \
  --object_store_memory 4294967296 \
  --seed 0 \
  --amp_type 'torch_amp' \
  --p_mcts_num 4 \
  --use_priority \
  --use_max_priority \
  --info 'EfficientZero-V1'