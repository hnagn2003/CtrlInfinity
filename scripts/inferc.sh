#!/bin/bash

# set arguments for inference
pn=0.06M
model_type=infinity_2b
use_scale_schedule_embedding=0
use_bit_label=1
checkpoint_type='torch'
BASE_MODEL=weights/infinity_2b_reg.pth
infinity_model_path=version_0_trangnvt2_bs4_gpu1_lr_0.006/ar-ckpt-giter001K-ep1-iter252-last
vae_type=32
vae_path=weights/infinity_vae_d32_reg.pth
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
--model_path local_output_2_reproduce/${infinity_model_path}.pth \
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
--save_folder output_2/${infinity_model_path} \
--use_image_adapter 1 \
--condition_folder data/infinity_toy_data/condition_canny \
--base_model ${BASE_MODEL}
