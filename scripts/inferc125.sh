#!/bin/bash

# set arguments for inference
pn=0.06M
model_type=infinity_layer12
use_scale_schedule_embedding=0
use_bit_label=1
checkpoint_type='torch'
BASE_MODEL=weights/infinity_125M_256x256.pth
infinity_model_path=version_0_hainn8_bs32_gpu1_baselr1e-4/ar-ckpt-giter010K-ep33-iter204-last
vae_type=16
vae_path=weights/infinity_vae_d16.pth
cfg=4
tau=0.5
rope2d_normalized_by_hw=2
add_lvl_embeding_only_first_block=1
rope2d_each_sa_layer=1
text_encoder_ckpt=google/flan-t5-xl
text_channels=2048
apply_spatial_patchify=0

# run inference
DESIGN_VERSION=0 python3 -m tools.run_infinity \
--cfg ${cfg} \
--tau ${tau} \
--pn ${pn} \
--model_path local_output125/${infinity_model_path}.pth \
--vae_type ${vae_type} \
--vae_path ${vae_path} \
--add_lvl_embeding_only_first_block ${add_lvl_embeding_only_first_block} \
--use_bit_label ${use_bit_label} \
--model_type ${model_type} \
--rope2d_each_sa_layer ${rope2d_each_sa_layer} \
--rope2d_normalized_by_hw ${rope2d_normalized_by_hw} \
--use_scale_schedule_embedding ${use_scale_schedule_embedding} \
--cfg ${cfg} \
--tau ${tau} \
--checkpoint_type ${checkpoint_type} \
--text_encoder_ckpt ${text_encoder_ckpt} \
--text_channels ${text_channels} \
--apply_spatial_patchify ${apply_spatial_patchify} \
--seed 1 \
--save_folder output125/${infinity_model_path} \
--use_image_adapter 1 \
--base_model ${BASE_MODEL} \
--condition_folder data/infinity_toy_data/condition_canny \
