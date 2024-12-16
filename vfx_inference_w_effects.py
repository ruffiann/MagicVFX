import torch
import argparse, os, cv2
import datetime, time
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from scripts.evaluation.funcs import batch_ddim_sampling
from scripts.evaluation.funcs import load_model_checkpoint, load_prompts, load_video_batch, get_filelist, save_videos
from utils.utils import instantiate_from_config
from lvdm.models.samplers.ddim import DDIMSampler
import torch.nn.functional as F
import nulltext_inversion

uc_embedding_list = None

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bg_path", type=str, default="demo/base_w_effect", help="file path of background videos")
    parser.add_argument("--boundingbox_path", type=str, default=None, help="file path of background boundingbox videos")
    parser.add_argument("--latent_mask", type=bool, default=False, help="mask for latent")
    parser.add_argument("--attention_mask", type=bool, default=False, help="mask for attention")
    parser.add_argument("--noise_level", type=float, default='0.5', help="noise_level")
    parser.add_argument("--seed", type=int, default=710, help="seed for seed_everything")
    parser.add_argument("--mode", default="base", type=str, help="which kind of inference mode: {'base', 'i2v'}")
    parser.add_argument("--ckpt_path", type=str, default='checkpoints/base_512_v2/model.ckpt', help="checkpoint path")
    parser.add_argument("--config", type=str, default='configs/inference_t2v_512_v2.0.yaml', help="config (yaml) path")
    parser.add_argument("--prompt_file", type=str, default='demo/prompt.txt', help="a text file containing many prompts")
    parser.add_argument("--neg_prompt_file", type=str, default=None, help="a text file containing many prompts")
    parser.add_argument("--savedir", type=str, default="final_results/w_effect", help="results saving path")
    parser.add_argument("--savefps", type=str, default=10, help="video fps to generate")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt",)
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM",)
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)",)
    parser.add_argument("--bs", type=int, default=1, help="batch size for inference")
    parser.add_argument("--height", type=int, default=320, help="image height, in pixel space")
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--frames", type=int, default=-1, help="frames num to inference")
    parser.add_argument("--fps", type=int, default=28)
    parser.add_argument("--unconditional_guidance_scale", type=float, default=12.0, help="prompt classifier-free guidance") #就是CFG
    parser.add_argument("--unconditional_guidance_scale_temporal", type=float, default=None, help="temporal consistency guidance")
    ## for conditional i2v only
    parser.add_argument("--cond_input", type=str, default=None, help="data dir of conditional input")
    return parser

def resize_video_as(video_tensor_to_resize, target_video_tensor):
    # Resize a video tensor to match the shape of another video tensor.
    target_depth = target_video_tensor.size(2)
    target_height = target_video_tensor.size(3)
    target_width = target_video_tensor.size(4)
    
    # resize
    resized_video_tensor = F.interpolate(video_tensor_to_resize, size=(target_depth, target_height, target_width), mode='nearest')

    resized_video_tensor = torch.cat((resized_video_tensor, resized_video_tensor[:, 0:1, :, :, :]), dim=1)
    
    return resized_video_tensor

def batch_ddim_sample(model, sampler, cond, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1.0,\
                        cfg_scale=1.0, temporal_cfg_scale=None, x_T=None, start_timesteps = None,\
                              x0=None, mask = None, attn_mask = None, neg_prompts=None, opt_uncond=False, **kwargs):
    # ddim_sampler = DDIMSampler(model)
    ddim_sampler = sampler
    uncond_type = model.uncond_type
    batch_size = noise_shape[0]

    ## construct unconditional guidance
    global uc_embedding_list
    if opt_uncond:
        if uc_embedding_list:
            uncond_embeddings_list = uc_embedding_list
        else:
            uncond_embeddings_list = nulltext_inversion.get_optimized_unconditional_embedding(model, sampler, x0, cond, ddim_steps, cfg_scale)
            uc_embedding_list = uncond_embeddings_list
    else:
        uncond_embeddings_list = None

    if cfg_scale != 1.0:
        if(neg_prompts):
            uc_emb = model.get_learned_conditioning(neg_prompts)
        elif uncond_type == "empty_seq":
            prompts = batch_size * [""]
            #prompts = N * T * [""]  ## if is_imgbatch=True
            uc_emb = model.get_learned_conditioning(prompts)
        elif uncond_type == "zero_embed":
            c_emb = cond["c_crossattn"][0] if isinstance(cond, dict) else cond
            uc_emb = torch.zeros_like(c_emb)
                
        ## process image embedding token
        if hasattr(model, 'embedder'): 
            uc_img = torch.zeros(noise_shape[0],3,224,224).to(model.device)
            ## img: b c h w >> b l c
            uc_img = model.get_image_embeds(uc_img)
            uc_emb = torch.cat([uc_emb, uc_img], dim=1)
        
        if isinstance(cond, dict): 
            uc = {key:cond[key] for key in cond.keys()}
            uc.update({'c_crossattn': [uc_emb]})
        else:
            uc = uc_emb
    else:
        uc = None
    
    # x_T = None
    batch_variants = []
    #batch_variants1, batch_variants2 = [], []
    for _ in range(n_samples):
        if ddim_sampler is not None:
            kwargs.update({"clean_cond": True})
            samples, _ = ddim_sampler.sample(S=ddim_steps,
                                            conditioning=cond,
                                            batch_size=noise_shape[0],
                                            shape=noise_shape[1:],
                                            verbose=False,
                                            unconditional_guidance_scale=cfg_scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            temporal_length=noise_shape[2],
                                            conditional_guidance_scale_temporal=temporal_cfg_scale,
                                            x_T=x_T,
                                            start_timesteps=start_timesteps,
                                            x0=x0,
                                            mask = mask,
                                            attn_mask = attn_mask,
                                            uncond_embeddings_list = uncond_embeddings_list,
                                            **kwargs
                                            )
        ## reconstruct from latent to pixel space
        batch_images = model.decode_first_stage_2DAE(samples)
        batch_variants.append(batch_images)
    ## batch, <samples>, c, t, h, w
    batch_variants = torch.stack(batch_variants, dim=1)
    return batch_variants

 
 
def run_inference(args, gpu_num, gpu_no, **kwargs):
    ############# step 1: model config
    ## -----------------------------------------------------------------
    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())
    model = instantiate_from_config(model_config)
    model = model.cuda(gpu_no)

    assert os.path.exists(args.ckpt_path), f"Error: checkpoint [{args.ckpt_path}] Not Found!"
    model = load_model_checkpoint(model, args.ckpt_path)
    model.eval()

    ############# step 2: sampler initialize and noise initialization
    ## -----------------------------------------------------------------
    ddim_sampler = DDIMSampler(model)
    ddim_sampler.make_schedule(ddim_num_steps=args.ddim_steps, ddim_eta=args.ddim_eta)

    ## sample shape
    assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
    ## latent noise shape
    h, w = args.height // 8, args.width // 8
    frames = model.temporal_length if args.frames < 0 else args.frames
    channels = model.channels
    
    ## saving folders
    os.makedirs(args.savedir, exist_ok=True)

    ############# step 3: load data
    ## -----------------------------------------------------------------
    # load and encode base videos
    assert os.path.exists(args.bg_path), "Error: bg file NOT Found!"
    bg_list = [f.path for f in os.scandir(args.bg_path) if f.path.endswith('.mp4')]
    bg_list = sorted(bg_list)
    num_samples = len(bg_list)
    video_batch = load_video_batch(filepath_list=bg_list, frame_stride=3, video_size=(320,512), video_frames=28).cuda()
    filename_list_rank = [i.split('/')[-1].split('.')[0] for i in bg_list]
    with torch.no_grad():
        video_latent = model.encode_first_stage_2DAE(video_batch)

    # multi gpu
    samples_split = num_samples // gpu_num
    residual_tail = num_samples % gpu_num
    print(f'[rank:{gpu_no}] {samples_split}/{num_samples} samples loaded.')
    indices = list(range(samples_split*gpu_no, samples_split*(gpu_no+1)))
    if gpu_no == 0 and residual_tail != 0:
        indices = indices + list(range(num_samples-residual_tail, num_samples))
    

    # load prompt file
    if(args.prompt_file):
        assert os.path.exists(args.prompt_file), "Error: prompt file NOT Found!"
        prompt_list = load_prompts(args.prompt_file)
        assert len(prompt_list) == num_samples, f"Error: prompt ({len(prompt_list)}) NOT match base videos ({num_samples})!"
        prompt_list_rank = [prompt_list[i] for i in indices]
    else:
        prompt_list_rank = ["" for i in indices]

    # load neg_prompt file if need
    if(args.neg_prompt_file):
        assert os.path.exists(args.neg_prompt_file), "Error: negative prompt file NOT Found!"
        neg_prompt_list = load_prompts(args.neg_prompt_file)
        assert len(neg_prompt_list) == num_samples, f"Error: negative prompt ({len(neg_prompt_list)}) NOT match prompt ({num_samples})!"
        neg_prompt_list_rank = [neg_prompt_list[i] for i in indices]
    else:
        neg_prompt_list_rank = ["" for i in indices]

    # load boundingbox file if need
    if(args.boundingbox_path):
        assert os.path.exists(args.boundingbox_path), "Error: boundingbox file NOT Found!"
        boundingbox_list = [f.path for f in os.scandir(args.boundingbox_path) if f.path.endswith('.mp4')]
        boundingbox_list = sorted(boundingbox_list)
        boundingbox_batch = load_video_batch(filepath_list=boundingbox_list, frame_stride=4, video_size=(320,512), video_frames=28, is_mask=True).cuda()
        assert len(boundingbox_batch) == num_samples, f"Error: boundingbox conditional input ({len(boundingbox_batch)}) NOT match prompt ({num_samples})!"
        ## resize boundingbox video
        boundingbox_latent = resize_video_as(boundingbox_batch, video_latent)
        token_ids = [2, 7, 3, 4]
        del boundingbox_batch
        

    del video_batch
    torch.cuda.empty_cache()

    # inverse level
    inverse_level = float(args.noise_level) # Between 0 and 1, the larger the value, the more noise will be added
    

    ############# step 4: run over samples
    ## -----------------------------------------------------------------
    start = time.time()
    n_rounds = len(prompt_list_rank) // args.bs
    n_rounds = n_rounds+1 if len(prompt_list_rank) % args.bs != 0 else n_rounds
    for idx in range(0, n_rounds):
        print(f'[rank:{gpu_no}] batch-{idx+1} ({args.bs})x{args.n_samples} ...')
        idx_s = idx*args.bs
        idx_e = min(idx_s+args.bs, len(prompt_list_rank))
        batch_size = idx_e - idx_s
        filenames = filename_list_rank[idx_s:idx_e]
        noise_shape = [batch_size, channels, frames, h, w]
        fps = torch.tensor([args.fps]*batch_size).to(model.device).long()

        prompts = prompt_list_rank[idx_s:idx_e]
        neg_prompts = neg_prompt_list_rank[idx_s:idx_e]
        if isinstance(prompts, str):
            prompts = [prompts]
            neg_prompts = [neg_prompts]
        text_emb = model.get_learned_conditioning(prompts)

        if args.mode == 'base':
            cond = {"c_crossattn": [text_emb], "fps": fps} #
        else:
            raise NotImplementedError

        # initial noise
        x0 = video_latent[idx].unsqueeze(0)

        if(args.latent_mask):
            mask = 1. - boundingbox_latent[idx].unsqueeze(0)
        else:
            mask = None

        if(args.attention_mask):
            attn_mask = boundingbox_latent[idx].unsqueeze(0)
            attn_mask[attn_mask>0] = 0.9
            token_id = token_ids[idx]
            attn_mask[0,0,0,0,0] += token_id
        else:
            attn_mask = None
        

        ## diffusion
        batch_samples = batch_ddim_sample(model, ddim_sampler, cond, noise_shape, args.n_samples, \
                                                args.ddim_steps, args.ddim_eta, args.unconditional_guidance_scale,
                                                # x_T=x_T, 
                                                start_timesteps=inverse_level, x0=x0, 
                                                mask = mask,
                                                attn_mask = attn_mask,
                                                neg_prompts = neg_prompts,
                                                **kwargs)

        ## b,samples,c,t,h,w
        save_videos(batch_samples, args.savedir, filenames, fps=args.savefps) # save the video

    print(f"Saved in {args.savedir}. Time used: {(time.time() - start):.2f} seconds")

if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("@CoLVDM Inference: %s"%now)
    parser = get_parser()
    args = parser.parse_args()
    seed_everything(args.seed)
    rank, gpu_num = 0, 1
    run_inference(args, gpu_num, rank)

