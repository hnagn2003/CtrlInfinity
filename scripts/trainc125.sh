#!/usr/bin/env bash

set -x

# set dist args
# select an avalable port
init_port=12345
while true; do
  if ! nc -z 127.0.0.1 $init_port; then
    break
  fi
  init_port=$((init_port + 1))
done
SINGLE=0
ARNOLD_WORKER_NUM=1
ARNOLD_ID=0
BATCH_SIZE=32
base_pretrained_path=weights/infinity_125M_256x256.pth
adapter_pretrained_path='abc'
eval_vae_path=weights/vae16.pth
eval_vae_config=weights/vae16.json
learning_rate=1e-4
val_log_freq=100
NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [ -z "$DESIGN_VERSION" ]; then
  DESIGN_VERSION=0
else
  DESIGN_VERSION=${DESIGN_VERSION}
fi
# global lr = lr * bs * num_gpus
# global_lr=$((BATCH_SIZE * NUM_GPUS * learning_rate))

exp_name=version_${DESIGN_VERSION}_${USER}_bs${BATCH_SIZE}_gpu${NUM_GPUS}_baselr${learning_rate}

if [ ! -z "$SINGLE" ] && [ "$SINGLE" != "0" ]; then
  echo "[single node alone] SINGLE=$SINGLE"
  nnodes=1
  node_rank=0
  nproc_per_node=1
  master_addr=127.0.0.1
  master_port=12347
else
  MASTER_NODE_ID=0
  nnodes=${ARNOLD_WORKER_NUM}
  node_rank=${ARNOLD_ID}
  master_addr="METIS_WORKER_${MASTER_NODE_ID}_HOST"
  master_addr=${!master_addr}
  master_port="METIS_WORKER_${MASTER_NODE_ID}_PORT"
  master_port=${!master_port}
  ports=(`echo $master_port | tr ',' ' '`)
  master_port=${ports[0]}
fi
master_addr=127.0.0.1
master_port=${init_port}
nproc_per_node=${NUM_GPUS}
echo "[nproc_per_node: ${nproc_per_node}]"
echo "[nnodes: ${nnodes}]"
echo "[node_rank: ${node_rank}]"
echo "[master_addr: ${master_addr}]"
echo "[master_port: ${master_port}]"

# set up envs
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3


BED=checkpoints125
LOCAL_OUT=local_output125
mkdir -p $BED
mkdir -p $LOCAL_OUT

export COMPILE_GAN=0
export USE_TIMELINE_SDK=1
export CUDA_TIMER_STREAM_KAFKA_CLUSTER=bmq_data_va
export CUDA_TIMER_STREAM_KAFKA_TOPIC=megatron_cuda_timer_tracing_original_v2
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

wandb disabled
bed_path=${BED}/${exp_name}/
data_path='../RepControlNet/data/canny_laion/infinity_10k/splits'
video_data_path=''
local_out_path=${LOCAL_OUT}/${exp_name}

DESIGN_VERSION=${DESIGN_VERSION} torchrun \
--nproc_per_node=${nproc_per_node} \
--nnodes=${nnodes} \
--node_rank=${node_rank} \
--master_addr=${master_addr} \
--master_port=${master_port} \
train.py \
--ep=100 \
--opt=adamw \
--cum=3 \
--sche=lin0 \
--fp16=2 \
--ada=0.9_0.97 \
--tini=-1 \
--tclip=5 \
--flash=0 \
--alng=5e-06 \
--saln=1 \
--cos=1 \
--enable_checkpointing=full-block \
--local_out_path ${local_out_path} \
--task_type='t2i' \
--bed=${bed_path} \
--data_path=${data_path} \
--video_data_path=${video_data_path} \
--exp_name=${exp_name} \
--pn 0.06M \
--model=layer12c4 \
--lbs ${BATCH_SIZE} \
--workers=4 \
--short_cap_prob 0.5 \
--online_t5=1 \
--use_streaming_dataset 1 \
--iterable_data_buffersize 30000 \
--Ct5=2048 \
--t5_path=google/flan-t5-xl \
--vae_type 16 \
--vae_ckpt=weights/infinity_vae_d16.pth  \
--wp 0.00000001 \
--wpe=1 \
--dynamic_resolution_across_gpus 1 \
--enable_dynamic_length_prompt 1 \
--reweight_loss_by_scale 1 \
--add_lvl_embeding_only_first_block 1 \
--rope2d_each_sa_layer 1 \
--rope2d_normalized_by_hw 2 \
--use_fsdp_model_ema 0 \
--always_training_scales 100 \
--use_bit_label 1 \
--zero=2 \
--save_model_iters_freq 500 \
--log_freq=${val_log_freq} \
--checkpoint_type='torch' \
--prefetch_factor=16 \
--noise_apply_strength 0.3 \
--noise_apply_layers 13 \
--apply_spatial_patchify 0 \
--use_flex_attn=False \
--pad=128 \
--use_image_adapter=1 \
--condition_folder=../RepControlNet/data/canny_laion/infinity_10k/condition_canny \
--image_folder=../RepControlNet/data/canny_laion/infinity_10k/images \
--pretrained_path ${base_pretrained_path} \
--adapter_pretrained_path ${adapter_pretrained_path} \
--tblr=${learning_rate} \
--validation True \
--eval_vae_path ${eval_vae_path} \
--eval_vae_config ${eval_vae_config} \
--auto_resume False