import gc
import json
import math
import os
import random
import sys
import time
import traceback
from collections import deque
from contextlib import nullcontext
from functools import partial
from distutils.util import strtobool
from typing import List, Optional, Tuple
import numpy as np
import torch
from torch.nn import functional as F
from torch.profiler import record_function
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5EncoderModel, T5TokenizerFast
import torch.distributed as tdist

import infinity.utils.dist as dist
from infinity.dataset.build import build_t2i_dataset
from infinity.utils.save_and_load import CKPTSaver, auto_resume
from infinity.utils import arg_util, misc, wandb_utils
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w
from tools.run_infinity import gen_one_img
import cv2
enable_timeline_sdk = False

def log_validation(infinity, vae, text_tokenizer, text_encoder):
    # generate validation images
    condition_folder = "data/infinity_toy_data/condition_canny"
    print(" <<<<<<<<<<<<<<<<<<<<<<<<<< start validation")
    prompts = [
        {'prompt': 'A rustic-style bar setup with a unique, antique-inspired design. The focal point is a large wooden wheel, likely from a wagon or cart, which has been repurposed as the central support for the bar. The wheel has a thick, dark wooden hub and several large spokes connecting it to the outer rim. The wooden surface of the bar is wide and flat, with a smooth finish, resting on the wheel for support.\n\nOn top of the bar, there are several items arranged neatly. There is a white bowl filled with yellow lemons, a vase containing several lit candles, and a couple of wine glasses, suggesting a cozy and inviting atmosphere. The lighting appears to be soft and warm, enhancing the rustic charm of the scene.\n\nTo the left and right of the bar are two matching stools. These stools have a simple, sturdy design with round seats and four legs. The wood on the stools appears to be of the same material and finish as the bar, contributing to the cohesive aesthetic of the setup.\n\nIn the background, there is a glimpse of a room with a cream-colored wall and a framed piece of artwork. The wall and artwork add to the overall warm and inviting ambiance of the space. The lighting in the background is soft and diffused, creating a pleasant bokeh effect that highlights the main elements of the bar setup while keeping the background details subtle.\n\nThe style of the image is photographic, with a standard lens style that captures the scene in a clear and detailed manner. The focus is sharp on the main objects, while the background is softly blurred, drawing attention to the rustic bar and its decorative elements. The warm lighting and earthy tones contribute to a cozy, inviting atmosphere, making the scene feel both functional and aesthetically pleasing.', 'image_path': '4265467520443567280.jpg'}, 
        {'prompt': 'a rustic bar table with stools and a wagon wheel', 'image_path': '4265467520443567280.jpg'},
        {'prompt': "A digital art style representation of a phone case design. It features three cartoon animals stacked on top of each other against a light blue background with small paw print patterns. \n\nFrom bottom to top:\n1. The bottom animal is a white polar bear with black eyes and ears. It is lying down with its front paws stretched out in front and appears relaxed.\n2. The middle animal is a black and white panda. It is sitting on the polar bear's back, with its front paws resting on the polar bear's back and its hind legs hanging down.\n3. The top animal is a brown bear. It is standing on the panda's back, with its front paws raised and a cheerful expression on its face.\n\nThe image has a clean and simple design with a clear focus on the cartoon animals. The style is cute and whimsical, aimed at creating a friendly and playful aesthetic. The phone case has rounded corners and a standard camera cut-out on the top left corner.", 'image_path': '6128985124434332020.jpg'},
        {'prompt': "a phone case with two bears and a panda on it", 'image_path': '6128985124434332020.jpg'}
        ]
    prompts = [{'prompt': 'a rustic bar table with stools and a wagon wheel', 'image_path': '4265467520443567280.jpg'}]

    val_scale_schedule = dynamic_resolution_h_w[1.0]["0.06M"]['scales']
    val_scale_schedule = [ (1, h, w) for (_, h, w) in val_scale_schedule]
    generated_images = []
    for sample in prompts:
        prompt = sample['prompt']
        condition_path = os.path.join(condition_folder, sample['image_path'])
        with torch.no_grad():
            generated_image = gen_one_img(
                infinity,
                vae,
                text_tokenizer,
                text_encoder,
                prompt,
                gt_leak=0,
                gt_ls_Bl=None,
                cfg_list=4.0,
                tau_list=0.5,
                scale_schedule=val_scale_schedule,
                cfg_insertion_layer=[0],
                vae_type=32,
                sampling_per_bits=1,
                enable_positive_prompt=0,
                eval_mode=True,
                condition_path=condition_path,
            ).cpu()
        generated_images.append(generated_image)
    print(" <<<<<<<<<<<<<<<<<<<<<<<<<< end validation")

    torch.cuda.empty_cache()
    return generated_images
def build_everything_from_args(args: arg_util.Args, saver):
    # set seed
    args.set_initial_seed(benchmark=True)
    if args.seed is not None and not args.rand: # check the randomness
        misc.check_randomness(args)

    # build data
    iters_train, ld_train, ld_val = build_dataloaders(args)   
    train_h_div_w_list = list(ld_train.dataset.h_div_w_template2generator.keys())
    print(f"{train_h_div_w_list=}")
    args.train_h_div_w_list = train_h_div_w_list 

    # load VAE
    print(f'Load vae form {args.vae_ckpt}')
    import os
    if not os.path.exists(args.vae_ckpt):
        vae_ckpt = {}
    else:
        vae_ckpt = torch.load(args.vae_ckpt, map_location='cpu')

    # build models. Note that here gpt is the causal VAR transformer which performs next scale prediciton with text guidance
    text_tokenizer, text_encoder, vae_local, gpt_uncompiled, gpt_wo_ddp, gpt_ddp, gpt_wo_ddp_ema, gpt_ddp_ema, gpt_optim = build_model_optimizer(args, vae_ckpt)
    
    # IMPORTANT: import heavy package `InfinityTrainer` after the Dataloader object creation/iteration to avoid OOM
    import os
    des_ver = os.environ["DESIGN_VERSION"]
    if des_ver == "1":
        from trainer1 import InfinityTrainer
    elif des_ver == "2":
        from trainer2 import InfinityTrainer
    elif des_ver == "3":
        from trainer3 import InfinityTrainer
    elif des_ver == "4":
        from trainer4 import InfinityTrainer
    else:
        print(" >>>>>>>>>>>>>>>> using default trainer")
        from trainer import InfinityTrainer
    # build trainer
    trainer = InfinityTrainer(
        is_visualizer=dist.is_visualizer(), device=args.device, raw_scale_schedule=args.scale_schedule, resos=args.resos,
        vae_local=vae_local, gpt_wo_ddp=gpt_wo_ddp, gpt=gpt_ddp, ema_ratio=args.tema, max_it=iters_train * args.ep,
        gpt_opt=gpt_optim, label_smooth=args.ls, z_loss_ratio=args.lz, eq_loss=args.eq, xen=args.xen,
        dbg_unused=args.dbg, zero=args.zero, vae_type=args.vae_type,
        reweight_loss_by_scale=args.reweight_loss_by_scale, gpt_wo_ddp_ema=gpt_wo_ddp_ema, 
        gpt_ema=gpt_ddp_ema, use_fsdp_model_ema=args.use_fsdp_model_ema, other_args=args,
    )
    
    # auto resume from broken experiment
    auto_resume_info, start_ep, start_it, acc_str, eval_milestone, trainer_state, args_state = auto_resume(args, 'ar-ckpt*.pth')
    print(f'global bs={args.glb_batch_size}, local bs={args.batch_size}')
    print(f'initial args:\n{str(args)}')
    args.dump_log()
    if start_ep == args.ep:
        args.dump_log()
        print(f'[vgpt] AR finished ({acc_str}), skipping ...\n\n')
        return None
    if trainer_state is not None and len(trainer_state):
        trainer.load_state_dict(trainer_state, strict=False, skip_vae=True) # don't load vae again
    
    start_it = start_it % iters_train
    print(f"{start_it=}, {iters_train=}")
    
    del vae_local, gpt_uncompiled, gpt_wo_ddp, gpt_ddp, gpt_wo_ddp_ema, gpt_ddp_ema, gpt_optim
    dist.barrier()
    return (
        text_tokenizer, text_encoder, trainer,
        start_ep, start_it, acc_str, eval_milestone, iters_train, ld_train, ld_val
    )


def build_model_optimizer(args, vae_ckpt):
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    import os
    des_ver = os.environ["DESIGN_VERSION"]
    if des_ver == "1":
        from infinity.models.infinity1 import MultipleLayers
        print(" >>>>>>>>>>>>>>>> using infinity design version 1 <<<<<<<<<<<<<<<<<<")
    elif des_ver == "2":
        from infinity.models.infinity2 import MultipleLayers
        print(" >>>>>>>>>>>>>>>> using infinity design version 2 <<<<<<<<<<<<<<<<<<")
    elif des_ver == "3":
        from infinity.models.infinity3 import MultipleLayers
        print(" >>>>>>>>>>>>>>>> using infinity design version 3 <<<<<<<<<<<<<<<<<<")
    elif des_ver == "4":
        from infinity.models.infinity4 import MultipleLayers
        print(" >>>>>>>>>>>>>>>> using infinity design version 4 <<<<<<<<<<<<<<<<<<")
    else:
        print(" >>>>>>>>>>>>>>>> using default trainer")
        from infinity.models.infinity import MultipleLayers
    from infinity.models.init_param import init_weights
    from infinity.utils.amp_opt import AmpOptimizer
    from infinity.utils.lr_control import filter_params
    from infinity.utils.load import build_vae_gpt
    
    # disable builtin initialization for speed
    setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
    setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
    vae_local, gpt_wo_ddp, gpt_wo_ddp_ema = build_vae_gpt(args, vae_ckpt, skip_gpt=False, device=args.model_init_device)
    del vae_ckpt
    if args.tini < 0:
        args.tini = math.sqrt(1 / gpt_wo_ddp.C / 3)
    init_weights(gpt_wo_ddp, other_std=args.tini)
    gpt_wo_ddp.special_init(aln_init=args.aln, aln_gamma_init=args.alng, scale_head=args.hd0, scale_proj=args.diva)

    if args.rush_resume:
        print(f"{args.rush_resume=}")
        cpu_d = torch.load(args.rush_resume, 'cpu')
        if 'trainer' in cpu_d:
            state_dict = cpu_d['trainer']['gpt_fsdp']
            ema_state_dict = cpu_d['trainer'].get('gpt_ema_fsdp', state_dict)
        else:
            state_dict = cpu_d
            ema_state_dict = state_dict
        def drop_unfit_weights(state_dict):
            if 'word_embed.weight' in state_dict and (state_dict['word_embed.weight'].shape[1] != gpt_wo_ddp.word_embed.in_features):
                del state_dict['word_embed.weight']
            if 'head.weight' in state_dict and (state_dict['head.weight'].shape[0] != gpt_wo_ddp.head.out_features):
                del state_dict['head.weight']
            if 'head.bias' in state_dict and (state_dict['head.bias'].shape[0] != gpt_wo_ddp.head.bias.shape[0]):
                del state_dict['head.bias']
            if state_dict['text_proj_for_sos.ca.mat_kv.weight'].shape != gpt_wo_ddp.text_proj_for_sos.ca.mat_kv.weight.shape:
                del state_dict['cfg_uncond']
                for key in list(state_dict.keys()):
                    if 'text' in key:
                        del state_dict[key]
            return state_dict
        
        gpt_wo_ddp.load_state_dict(drop_unfit_weights(state_dict), strict=False)
        if args.use_fsdp_model_ema:
            gpt_wo_ddp_ema.load_state_dict(drop_unfit_weights(ema_state_dict), strict=False)

    if args.rwe:
        gpt_wo_ddp.word_embed.weight.requires_grad = False
        torch.nn.init.trunc_normal_(gpt_wo_ddp.word_embed.weight.data, std=1.5 * math.sqrt(1 / gpt_wo_ddp.C / 3))
        if hasattr(gpt_wo_ddp.word_embed, 'bias'):
            gpt_wo_ddp.word_embed.bias.requires_grad = False
            gpt_wo_ddp.word_embed.bias.data.zero_()
    # if args.pretrained_path:
    #     print(f'>>>>>>>>>>>>>> Load pretrained model from {args.pretrained_path} <<<<<<<<<<<<<<<<<')
    #     ckpt = torch.load(args.pretrained_path, map_location='cpu')
         
    #     # check if ckpt has "trainer" key
    #     if "trainer" in ckpt:
    #         state_dict = ckpt['trainer']['gpt_fsdp']
    #     else:
    #         state_dict = ckpt
    # if args.adapter_pretrained_path is not None and os.path.exists(args.adapter_pretrained_path):
    #     adapter_state_dict = torch.load(args.adapter_pretrained_path, map_location='cpu')
    #     state_dict.update(adapter_state_dict)
    # gpt_wo_ddp.load_state_dict(state_dict, strict=False)
        
    ndim_dict = {name: para.ndim for name, para in gpt_wo_ddp.named_parameters() if para.requires_grad}
    print(f'[PT] GPT model = {gpt_wo_ddp}\n\n')
    count_p = lambda m: f'{sum(p.numel() for p in m.parameters()) / 1e6:.2f}'
    print(f'[PT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (
        ('VAE', vae_local), ('VAE.quant', vae_local.quantize)
    )]))
    print(f'[PT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (
        ('GPT', gpt_wo_ddp),
    )]) + '\n\n')
    
    gpt_uncompiled = gpt_wo_ddp
    gpt_wo_ddp = args.compile_model(gpt_wo_ddp, args.tfast)

    gpt_ddp_ema = None
    if args.zero:
        from torch.distributed.fsdp import ShardingStrategy
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy
        from torch.distributed.device_mesh import init_device_mesh

        # use mix prec: https://github.com/pytorch/pytorch/issues/76607
        if gpt_wo_ddp.num_block_chunks == 1:  # no chunks
            auto_wrap_policy = ModuleWrapPolicy([type(gpt_wo_ddp.unregistered_blocks[0]), ])
        else:
            auto_wrap_policy = ModuleWrapPolicy([MultipleLayers, ])
        
        if args.enable_hybrid_shard:
            sharding_strategy = ShardingStrategy.HYBRID_SHARD if args.zero == 3 else ShardingStrategy._HYBRID_SHARD_ZERO2
            world_size = dist.get_world_size()
            assert world_size % args.inner_shard_degree == 0
            assert args.inner_shard_degree > 1 and args.inner_shard_degree < world_size
            device_mesh = init_device_mesh('cuda', (world_size // args.inner_shard_degree, args.inner_shard_degree))
        else:
            sharding_strategy = ShardingStrategy.FULL_SHARD if args.zero == 3 else ShardingStrategy.SHARD_GRAD_OP
            device_mesh = None
        print(f'{">" * 45 + " " * 5} FSDP INIT with {args.zero=} {sharding_strategy=} {auto_wrap_policy=} {" " * 5 + "<" * 45}', flush=True)
        
        gpt_ddp: FSDP = FSDP(
            gpt_wo_ddp, 
            device_id=dist.get_local_rank(),
            sharding_strategy=sharding_strategy, 
            mixed_precision=None,
            auto_wrap_policy=auto_wrap_policy, 
            use_orig_params=True, 
            sync_module_states=True, 
            limit_all_gathers=True,
            device_mesh=device_mesh,
        ).to(args.device)
        if args.use_fsdp_model_ema:
            gpt_wo_ddp_ema = gpt_wo_ddp_ema.to(args.device)
            gpt_ddp_ema: FSDP = FSDP(
                gpt_wo_ddp_ema, 
                device_id=dist.get_local_rank(),
                sharding_strategy=sharding_strategy, 
                mixed_precision=None,
                auto_wrap_policy=auto_wrap_policy, 
                use_orig_params=args.fsdp_orig, 
                sync_module_states=True, 
                limit_all_gathers=True,
            )
    else:
        ddp_class = DDP if dist.initialized() else misc.NullDDP
        gpt_ddp: DDP = ddp_class(gpt_wo_ddp, device_ids=[dist.get_local_rank()], find_unused_parameters=args.dbg, broadcast_buffers=False)
    torch.cuda.synchronize()

    # =============== build optimizer ===============
    nowd_keys = set()
    if args.nowd >= 1:
        nowd_keys |= {
            'cls_token', 'start_token', 'task_token', 'cfg_uncond',
            'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
            'gamma', 'beta',
            'ada_gss', 'moe_bias',
            'scale_mul',
            'text_proj_for_sos.ca.mat_q',
        }
    if args.nowd >= 2:
        nowd_keys |= {'class_emb', 'embedding'}
    names, paras, para_groups = filter_params(gpt_ddp if args.zero else gpt_wo_ddp, ndim_dict, nowd_keys=nowd_keys)
    del ndim_dict
    if '_' in args.ada:
        beta0, beta1 = map(float, args.ada.split('_'))
    else:
        beta0, beta1 = float(args.ada), -1
    
    opt_clz = {
        'sgd':   partial(torch.optim.SGD, momentum=beta0, nesterov=True),
        'adam':  partial(torch.optim.AdamW, betas=(beta0, beta1), fused=args.afuse),
        'adamw': partial(torch.optim.AdamW, betas=(beta0, beta1), fused=args.afuse),
    }[args.opt]
    opt_kw = dict(lr=args.tlr, weight_decay=0)
    if args.oeps: opt_kw['eps'] = args.oeps
    print(f'[vgpt] optim={opt_clz}, opt_kw={opt_kw}\n')
    gpt_optim = AmpOptimizer('gpt', args.fp16, opt_clz(params=para_groups, **opt_kw), gpt_ddp if args.zero else gpt_wo_ddp, args.r_accu, args.tclip, args.zero)
    del names, paras, para_groups
    
    if args.online_t5:
        print(f'Loading T5 from {args.t5_path}...')
        text_tokenizer: T5TokenizerFast = AutoTokenizer.from_pretrained(args.t5_path, revision=None, legacy=True)
        text_tokenizer.model_max_length = args.tlen
        text_encoder: T5EncoderModel = T5EncoderModel.from_pretrained(args.t5_path, torch_dtype=torch.float16)
        text_encoder.to(args.device)
        text_encoder.eval()
        text_encoder.requires_grad_(False)
        [p.requires_grad_(False) for p in text_encoder.parameters()]
    else:
        text_tokenizer = text_encoder = None
    
    return text_tokenizer, text_encoder, vae_local, gpt_uncompiled, gpt_wo_ddp, gpt_ddp, gpt_wo_ddp_ema, gpt_ddp_ema, gpt_optim

def build_dataloaders(args):
    if args.task_type == 't2i':
        dataset_train = build_t2i_dataset(
            args, 
            args.data_path, 
            args.data_load_reso, 
            max_caption_len=args.tlen, 
            short_prob=args.short_cap_prob, 
            load_vae_instead_of_image=False,
            condition_folder=args.condition_folder,
            image_folder=args.image_folder,
        )
    else:
        raise NotImplementedError(f'args.task_type={args.task_type} not supported')
    type_train_set = type(dataset_train).__name__
    vbs = round(args.batch_size * 1.5)
    print(f"{args.batch_size=}, {vbs=}", flush=True)
    ld_val = math.ceil(50000 / vbs)
    ld_train = DataLoader(dataset=dataset_train, num_workers=args.workers, pin_memory=True, generator=args.get_different_generator_for_each_rank(), batch_size=None, prefetch_factor=args.prefetch_factor)
    iters_train = len(ld_train)
    print(f'len(dataloader): {len(ld_train)}, len(dataset): {len(dataset_train)}, total_samples: {dataset_train.total_samples()}')
    print(f'[dataloader] gbs={args.glb_batch_size}, lbs={args.batch_size}, iters_train={iters_train}, type(train_set)={type_train_set}')
    return iters_train, ld_train, ld_val

def load_eval_vae(vae_path, vae_config):
    vae_config = json.load(open(vae_config))
    import argparse

    args = argparse.Namespace(**vae_config)
    from infinity.models.bsq_vae.flux_vqgan import AutoEncoder
    vae = AutoEncoder(args)
    vae.load_state_dict(torch.load(vae_path), strict=True)
    vae.eval()
    vae = vae.to('cuda')
    return vae

def main_train(args: arg_util.Args):
    saver = CKPTSaver(dist.is_master(), eval_milestone=None)
    ret = build_everything_from_args(args, saver)
    
    if ret is None:
        return
    
    (
        text_tokenizer, text_encoder, trainer,
        start_ep, start_it, acc_str, eval_milestone,
        iters_train, ld_train, ld_val
    ) = ret
    gc.collect(), torch.cuda.empty_cache()
    
    # import heavy packages after Dataloader object creation
    import os
    des_ver = os.environ["DESIGN_VERSION"]
    if des_ver == "1":
        from trainer1 import InfinityTrainer
    elif des_ver == "2":
        from trainer2 import InfinityTrainer
    elif des_ver == "3":
        from trainer3 import InfinityTrainer
    elif des_ver == "4":
        from trainer4 import InfinityTrainer
    else:
        print(" >>>>>>>>>>>>>>>> using default trainer")
        from trainer import InfinityTrainer
    ret: Tuple[
        misc.TensorboardLogger, T5TokenizerFast, T5EncoderModel, InfinityTrainer,
        int, int, str, List[Tuple[float, float]], Optional[int], Optional[DataLoader], DataLoader,
    ]
    if args.validation:
        eval_vae = load_eval_vae(args.eval_vae_path, args.eval_vae_config)
    else:
        eval_vae = None
    world_size = int(os.environ["WORLD_SIZE"])
    start_time, min_L_mean, min_L_tail, max_acc_mean, max_acc_tail = time.time(), 999., 999., -1., -1.
    last_val_loss_mean, best_val_loss_mean, last_val_acc_mean, best_val_acc_mean = 999., 999., 0., 0.
    last_val_loss_tail, best_val_loss_tail, last_val_acc_tail, best_val_acc_tail = 999., 999., 0., 0.
    seg5 = np.linspace(1, args.ep, 5+1, dtype=int).tolist()
    logging_params_milestone: List[int] = np.linspace(1, args.ep, 10+1, dtype=int).tolist()
    milestone_ep_feishu_log = set(seg5[:])
    vis_milestone_ep = set(seg5[:]) | set(x for x in (2, 4, 8, 16) if x <= args.ep)
    for x in [6, 12, 3, 24, 18, 48, 72, 96]:
        if len(vis_milestone_ep) < 10 and x <= args.ep:
            vis_milestone_ep.add(x)
    
    PARA_EMB, PARA_ALN, PARA_OT = 0, 0, 0
    for n, p in trainer.gpt_wo_ddp.named_parameters():
        if not p.requires_grad: continue
        if any(k in n for k in ('class_emb', 'pos_1LC', 'lvl_embed')):
            PARA_EMB += p.numel()
        elif any(k in n for k in ('ada_lin',)):
            PARA_ALN += p.numel()
        else:
            PARA_OT += p.numel()
    PARA_ALL = PARA_EMB + PARA_ALN + PARA_OT
    
    trainer.gpt_opt.log_param(ep=-1)
    time.sleep(3), gc.collect(), torch.cuda.empty_cache(), time.sleep(3)
    ep_lg = max(1, args.ep // 10) if args.ep <= 100 else max(1, args.ep // 20)
    
    # ============================================= epoch loop begins =============================================
    L_mean, L_tail = -1, -1
    epochs_loss_nan = 0
    # # print all trainable parameters
    # for name, param in trainer.gpt_wo_ddp.named_parameters():
    #     if param.requires_grad:
    #         print(f'{name}: {param.shape}')
    # build wandb logger
    if dist.is_master():
        wandb_utils.wandb.init(project=args.project_name, name=args.exp_name, config={})
    os.makedirs(os.path.join(args.local_out_path, 'validation'), exist_ok=True)
    for ep in range(start_ep, args.ep):
        if ep % ep_lg == 0 or ep == start_ep:
            print(f'[PT info]  from ep{start_ep} it{start_it}, acc_str: {acc_str}, diffs: {args.diffs},    =======>  bed: {args.bed}  <=======\n')
        # set epoch for dataloader
        if args.use_streaming_dataset:
            ld_train.dataset.set_epoch(ep)
        # [train one epoch]
        stats, (sec, remain_time, finish_time) = train_one_ep(
            ep=ep,
            is_first_ep=ep == start_ep,
            start_it=start_it if ep == start_ep else 0,
            me=None,
            saver=saver,
            args=args,
            ld_or_itrt=iter(ld_train),
            iters_train=iters_train,
            text_tokenizer=text_tokenizer, text_encoder=text_encoder,
            trainer=trainer,
            logging_params_milestone=logging_params_milestone,
            enable_timeline_sdk=enable_timeline_sdk,
            eval_vae=eval_vae
        )
        
        # [update the best loss or acc]
        L_mean, L_tail, acc_mean, acc_tail, grad_norm = stats['Lm'], stats['Lt'], stats['Accm'], stats['Acct'], stats['tnm']
        min_L_mean, max_acc_mean, max_acc_tail = min(min_L_mean, L_mean), max(max_acc_mean, acc_mean), max(max_acc_tail, acc_tail)
        if L_tail != -1:
            min_L_tail = min(min_L_tail, L_tail)
        
        # [check nan]
        epochs_loss_nan += int(not math.isfinite(L_mean))
        if (args.fp16 == 1 and epochs_loss_nan >= 2) or (args.fp16 != 1 and epochs_loss_nan >= 1):
            print(f'[rk{dist.get_rank():02d}] L_mean is {L_mean}, stopping training!', flush=True, force=True)
            sys.exit(666)
        
        # [logging]
        args.cur_phase = 'AR'
        args.cur_ep = f'{ep+1}/{args.ep}'
        args.remain_time, args.finish_time = remain_time, finish_time
        args.last_Lnll, args.last_Ld, args.acc_all, args.acc_real, args.acc_fake, args.last_wei_g = min_L_mean, min_L_tail, None, (None if max_acc_mean < 0 else max_acc_mean), (None if max_acc_tail < 0 else max_acc_tail), grad_norm
        if math.isfinite(args.last_wei_g) and args.last_wei_g > 4:
            args.grad_boom = 'boom'
        
        AR_ep_loss = {}
        is_val_and_also_saving = (ep + 1) % max(1, args.ep // 25) == 0 or (ep + 1) == args.ep
        if (ep + 1) < 10:
            law_stats = {
                'last_Lm': L_mean, 'best_Lm': min_L_mean, 'last_Am': acc_mean, 'best_Am': max_acc_mean,
                'last_Lt': L_tail, 'best_Lt': min_L_tail, 'last_At': acc_tail, 'best_At': max_acc_tail,
                'pe': PARA_EMB, 'paln': PARA_ALN, 'pot': PARA_OT, 'pall': PARA_ALL,
            }
        elif is_val_and_also_saving:
            if ld_val is None or isinstance(ld_val, int):    # args.nodata or args.nova
                last_val_loss_mean, last_val_loss_tail, last_val_acc_mean, last_val_acc_tail, tot, cost = 0.666, 0.555, 5.55, 6.66, 50000, 0.001
            else:
                last_val_loss_mean, last_val_loss_tail, last_val_acc_mean, last_val_acc_tail, tot, cost = trainer.eval_ep(ep, args, ld_val)
            
            best_val_loss_mean, best_val_loss_tail = min(best_val_loss_mean, last_val_loss_mean), min(best_val_loss_tail, last_val_loss_tail)
            best_val_acc_mean, best_val_acc_tail = max(best_val_acc_mean, last_val_acc_mean), max(best_val_acc_tail, last_val_acc_tail)
            AR_ep_loss['vL_mean'], AR_ep_loss['vL_tail'], AR_ep_loss['vacc_mean'], AR_ep_loss['vacc_tail'] = last_val_loss_mean, last_val_loss_tail, last_val_acc_mean, last_val_acc_tail
            print(f'  [*] [ep{ep}]  VAL {tot}  |  Lm: {L_mean:.4f}, Lt: {L_tail:.4f}, Accm: {acc_mean:.2f}, Acct: {acc_tail:.2f}, cost: {cost:.2f}s')
            law_stats = {
                'last_Lm': last_val_loss_mean, 'best_Lm': best_val_loss_mean, 'last_Am': last_val_acc_mean, 'best_Am': best_val_acc_mean,
                'last_Lt': last_val_loss_tail, 'best_Lt': best_val_loss_tail, 'last_At': last_val_acc_tail, 'best_At': best_val_acc_tail,
                'pe': PARA_EMB, 'paln': PARA_ALN, 'pot': PARA_OT, 'pall': PARA_ALL,
            }
        else: law_stats = None
        if dist.is_master() and law_stats is not None:
            stat_file = os.path.join(args.bed, 'law.stat')
            if os.path.exists(stat_file):
                with open(stat_file, 'r', encoding='utf-8') as law_fp: tag_to_epv = json.load(law_fp)
            else:
                tag_to_epv = {tag: {} for tag in law_stats.keys()}
            for tag, v in law_stats.items():
                tag_to_epv[tag][ep + 1] = v
            with open(stat_file, 'w', encoding='utf-8') as law_fp: json.dump(tag_to_epv, law_fp, indent=2)
            
            # ============= LEGACY =============
            with open(os.path.join(args.bed, 'law'), 'w') as law_fp:
                json.dump({
                    'last_Lm': last_val_loss_mean, 'best_Lm': best_val_loss_mean, 'last_Am': last_val_acc_mean, 'best_Am': best_val_acc_mean,
                    'last_Lt': last_val_loss_tail, 'best_Lt': best_val_loss_tail, 'last_At': last_val_acc_tail, 'best_At': best_val_acc_tail,
                    'pe': PARA_EMB, 'paln': PARA_ALN, 'pot': PARA_OT, 'pall': PARA_ALL,
                }, law_fp, indent=2)
        print(f'  [*] [ep{ep}]  Lmean: {min_L_mean:.3f} ({L_mean:.3f}), Ltail {min_L_tail:.3f} ({L_tail:.3f}),  Acc m-t: {max_acc_mean:.2f} {max_acc_tail:.2f},  Remain: {remain_time},  Finish: {finish_time}', flush=True)
        AR_ep_loss['L_mean'], AR_ep_loss['L_tail'], AR_ep_loss['acc_mean'], AR_ep_loss['acc_tail'] = L_mean, L_tail, acc_mean, acc_tail        
        args.dump_log()
    # ============================================= epoch loop ends =============================================
    
    total_time = f'{(time.time() - start_time) / 60 / 60:.1f}h'
    print('\n\n')
    print(f'  [*] [PT finished]  Total Time: {total_time},   Lm: {min_L_mean:.3f} ({L_mean}),   Lt: {min_L_tail:.3f} ({L_tail})')
    print('\n\n')
    
    del stats, iters_train, ld_train, visualizer
    time.sleep(3), gc.collect(), torch.cuda.empty_cache(), time.sleep(3)
    return


g_speed_ls = deque(maxlen=128)
def train_one_ep(
    ep: int, is_first_ep: bool, start_it: int, me: misc.MetricLogger,
    saver: CKPTSaver, args: arg_util.Args, ld_or_itrt, iters_train: int, 
    text_tokenizer: T5TokenizerFast, text_encoder: T5EncoderModel, trainer, logging_params_milestone, enable_timeline_sdk: bool, eval_vae
):
    import os
    # only run on main process
    
    # wrap back to fsdp
    # IMPORTANT: import heavy packages after the Dataloader object creation/iteration to avoid OOM
    import os
    des_ver = os.environ["DESIGN_VERSION"]
    if des_ver == "1":
        from trainer1 import InfinityTrainer
    elif des_ver == "2":
        from trainer2 import InfinityTrainer
    elif des_ver == "3":
        from trainer3 import InfinityTrainer
    elif des_ver == "4":
        from trainer4 import InfinityTrainer
    else:
        print(" >>>>>>>>>>>>>>>> using default trainer")
        from trainer import InfinityTrainer
    from infinity.utils.lr_control import lr_wd_annealing
    trainer: InfinityTrainer
    
    step_cnt = 0
    header = f'[Ep]: [{ep:4d}/{args.ep}]'
    with misc.Low_GPU_usage(files=[args.log_txt_path], sleep_secs=20, verbose=True) as telling_dont_kill:
        last_touch = time.time()
        g_it, max_it = ep * iters_train, args.ep * iters_train
        doing_profiling = args.prof and ep == 0 and (args.profall or dist.is_master())
        maybe_record_function = record_function if doing_profiling else nullcontext
        trainer.gpt_wo_ddp.maybe_record_function = maybe_record_function
        last_t_perf = time.time()
        speed_ls: deque = g_speed_ls
        FREQ = min(args.prof_freq, iters_train//2-1)
        NVIDIA_IT_PLUS_1 = set(FREQ*i for i in (1, 2, 3, 4, 6, 8))
        ranges = set([2 ** i for i in range(20)])
        if ep <= 1: ranges |= {1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40}
        PRINTABLE_IT_PLUS_1 = set(FREQ*i for i in ranges)

        me = misc.MetricLogger()
        [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{value:.2g}')) for x in ['tlr']]
        [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{median:.2f} ({global_avg:.2f})')) for x in ['tnm']]
        [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{median:.3f} ({global_avg:.3f})')) for x in ['Lm', 'Lt']]
        [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{median:.2f} ({global_avg:.2f})')) for x in ['Accm', 'Acct']]
        # ============================================= iteration loop begins =============================================
        for it, data in me.log_every(start_it, iters_train, ld_or_itrt, args.log_freq, args.log_every_iter, header):
            g_it = ep * iters_train + it

            # calling inc_step to sync the global_step
            if enable_timeline_sdk:
                ndtimeline.inc_step()

            if (it+1) % FREQ == 0:
                speed_ls.append((time.time() - last_t_perf) / FREQ)
                last_t_perf = time.time()

                if enable_timeline_sdk:
                    ndtimeline.flush()
            
            with torch.no_grad():
                if (g_it+1) % args.save_model_iters_freq == 0:
                    with misc.Low_GPU_usage(files=[args.log_txt_path], sleep_secs=3, verbose=True):
                        saver.sav(args=args, g_it=(g_it+1), next_ep=ep, next_it=it+1, trainer=trainer, acc_str=f'[todo]', eval_milestone=None, also_save_to=None, best_save_to=None)
            
            with maybe_record_function('before_train'):
                # [get data]
                inp, captions, condition = data
                tokens = text_tokenizer(text=captions, max_length=text_tokenizer.model_max_length, padding='max_length', truncation=True, return_tensors='pt')  # todo: put this into dataset
                input_ids = tokens.input_ids.cuda(non_blocking=True)
                mask = tokens.attention_mask.cuda(non_blocking=True)
                text_features = text_encoder(input_ids=input_ids, attention_mask=mask)['last_hidden_state'].float()
                
                lens: List[int] = mask.sum(dim=-1).tolist()
                cu_seqlens_k = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
                Ltext = max(lens)
                
                kv_compact = []
                for len_i, feat_i in zip(lens, text_features.unbind(0)):
                    kv_compact.append(feat_i[:len_i])
                kv_compact = torch.cat(kv_compact, dim=0)
                text_cond_tuple: Tuple[torch.FloatTensor, List[int], torch.LongTensor, int] = (kv_compact, lens, cu_seqlens_k, Ltext)
                inp = inp.to(args.device, non_blocking=True)
                if condition is not None:
                    condition = condition.to(args.device, non_blocking=True)
                if it > start_it + 10:
                    telling_dont_kill.early_stop()
                
                # [logging]
                args.cur_it = f'{it+1}/{iters_train}'
                args.last_wei_g = me.meters['tnm'].median
                if dist.is_local_master() and (it >= start_it + 10) and (time.time() - last_touch > 90):
                    _, args.remain_time, args.finish_time = me.iter_time.time_preds(max_it - g_it + (args.ep - ep) * 15)      # +15: other cost
                    args.dump_log()
                    last_touch = time.time()
                
                # [schedule learning rate]
                wp_it = args.wp * iters_train
                min_tlr, max_tlr, min_twd, max_twd = lr_wd_annealing(args.sche, trainer.gpt_opt.optimizer, args.tlr, args.twd, args.twde, g_it, wp_it, max_it, wp0=args.wp0, wpe=args.wpe)
                
                # [get scheduled hyperparameters]
                progress = g_it / (max_it - 1)
                clip_decay_ratio = (0.3 ** (20 * progress) + 0.2) if args.cdec else 1
                
                stepping = (g_it + 1) % args.ac == 0
                step_cnt += int(stepping)
            # if args.validation and g_it % args.log_freq == 0 and g_it > 10:
            #     if dist.is_master():
            # # unwrap from fsdp
            #         with torch.no_grad() and trainer.gpt.summon_full_params(trainer.gpt):
            #             val_images = log_validation(trainer.gpt, eval_vae, text_tokenizer, text_encoder)
            #                 # Clean up GPU memory after validation
            #                 # torch.cuda.empty_cache()
            #                 # gc.collect()
                            
            #             val_dir = os.path.join(args.local_out_path, f'validation/ep{ep}_it{it}.jpg')
            #             cv2.imwrite(val_dir, np.hstack(val_images))
            #             torch.cuda.empty_cache()
            #             gc.collect()


            with maybe_record_function('in_training'):
                grad_norm_t, scale_log2_t = trainer.train_step(
                    ep=ep, it=it, g_it=g_it, stepping=stepping, clip_decay_ratio=clip_decay_ratio,
                    metric_lg=me, 
                    logging_params=stepping and step_cnt == 1 and (ep < 4 or ep in logging_params_milestone), 
                    inp_B3HW=inp, 
                    text_cond_tuple=text_cond_tuple,
                    args=args, cond_B3HW=condition
                )
            
            with maybe_record_function('after_train'):
                me.update(tlr=max_tlr)
    # ============================================= iteration loop ends =============================================
    
    me.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in me.meters.items()}, me.iter_time.time_preds(max_it - (g_it + 1) + (args.ep - ep) * 15)  # +15: other cost


wait1 = os.path.join(os.path.expanduser('~'), 'wait1')
def main():     # # 'pt_le_ft' in train_vae.py is the same as 'pt_le_ft' in train_gpt.py
    if dist.is_local_master(): misc.os_system(f'touch {wait1}')
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    if args.validation:
        args.fused_norm = 1
    main_train(args)
    
    args.remain_time, args.finish_time = '-', time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() - 60))
    args.cur_phase = 'OK'
    print(f'final args:\n\n{str(args)}')
    args.dump_log()
    if isinstance(sys.stdout, dist.BackupStreamToFile) and isinstance(sys.stderr, dist.BackupStreamToFile):
        sys.stdout.close(), sys.stderr.close()
    if dist.is_local_master(): misc.os_system(f'rm -rf {wait1}')
    if args.vis and dist.is_visualizer():
        misc.os_system(f'hdfs dfs -get {args.tb_log_dir_online}/* {args.tb_log_dir}/ >/dev/null 2>&1')  # 'cp -r {args.local_out_path}/* {args.bed}/' is done by lockable.py or launch.py
    dist.barrier()
    time.sleep(120)


if __name__ == '__main__':
    try:
        main()
    except Exception as _e:
        time.sleep(dist.get_rank() * 1 + random.random() * 0.5)
        try:
            # noinspection PyArgumentList
            print(f'[rk{dist.get_rank():2d}] {type(_e).__name__}', flush=True, force=True)
        except:
            try: print(f'[rk{dist.get_rank():2d}] {type(_e).__name__}', flush=True)
            except: pass
        if dist.is_master():
            print(f'[err]:\n{_e}')
            traceback.print_exc()
        raise _e
    finally:
        misc.os_system(f'rm -rf {wait1}')
        dist.finalize()
        if isinstance(sys.stdout, dist.BackupStreamToFile) and isinstance(sys.stderr, dist.BackupStreamToFile):
            sys.stdout.close(), sys.stderr.close()
