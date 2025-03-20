from tools.run_infinity import load_visual_tokenizer
from infinity.dataset.dataset_t2i_iterable import transform
import glob
import os.path as osp
import numpy as np
import argparse
from PIL import Image
import torch
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
h_div_w = 1 / 1  # aspect ratio, height:width
h_div_w_template_ = h_div_w_templates[
    np.argmin(np.abs(h_div_w_templates - h_div_w))
]
dynamic_resolution_h_w, h_div_w_templates
scale_schedule = dynamic_resolution_h_w[h_div_w_template_]["1M"][
    "scales"
]
scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
from tools.run_infinity import joint_vi_vae_encode_decode
model_path = "weights/infinity_2b_reg.pth"
vae_path = "weights/infinity_vae_d32_reg.pth"
text_encoder_ckpt = "google/flan-t5-xl"
args = argparse.Namespace(
            pn="1M",
            model_path=model_path,
            cfg_insertion_layer=0,
            vae_type=32,
            vae_path=vae_path,
            add_lvl_embeding_only_first_block=1,
            use_bit_label=1,
            model_type="infinity_2b",
            rope2d_each_sa_layer=1,
            rope2d_normalized_by_hw=2,
            use_scale_schedule_embedding=0,
            sampling_per_bits=1,
            text_encoder_ckpt=text_encoder_ckpt,
            text_channels=2048,
            apply_spatial_patchify=0,
            h_div_w_template=1.000,
            use_flex_attn=0,
            cache_dir="/tmp/cache",
            checkpoint_type="torch",
            bf16=1,
        )
vae = load_visual_tokenizer(args)
# prepare data
data_path = "data/cannylaion/pose/"
image_paths = glob.glob(osp.join(data_path, "*.jpg"))
for image_path in image_paths:
    # image = Image.open(image_path).convert("RGB")
    # image = transform(image, 256, 256)
    # image = image.unsqueeze(0)
    # recons, vq_output = vae(image.cuda())
    # recons = recons.detach().cpu().numpy()
    # recons = np.transpose(recons, (0, 2, 3, 1))
    # recons = (recons * 255).astype(np.uint8)
    # recons = Image.fromarray(recons[0])
    # recons.save(f"local_output/{image_path.split('/')[-1]}")
    gt_img, recons_img, all_bit_indices = joint_vi_vae_encode_decode(vae, image_path, scale_schedule, "cuda", 256, 256)
        #save recons_img
    Image.fromarray(recons_img).save(f"local_output/recon_{image_path.split('/')[-1]}")
    Image.fromarray(gt_img).save(f"local_output/gt_{image_path.split('/')[-1]}")
    #save image
    