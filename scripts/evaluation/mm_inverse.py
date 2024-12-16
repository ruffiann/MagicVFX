import torch
import argparse, os, cv2
import datetime, time
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from funcs import batch_ddim_sampling
from funcs import load_model_checkpoint, load_prompts, load_video_batch, get_filelist, save_videos
from utils.utils import instantiate_from_config
from lvdm.models.samplers.ddim import DDIMSampler

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bg_path", type=str, default="/mnt/hdd3/guojiaqi/MM/input/video/", help="file path of background videos")
    # parser.add_argument("--fg_path", type=str, default="/mnt/hdd3/guojiaqi/MM/input/autumn", help="file path of background videos")
    parser.add_argument("--seed", type=int, default=20230211, help="seed for seed_everything")
    parser.add_argument("--mode", default="base", type=str, help="which kind of inference mode: {'base', 'i2v'}")
    parser.add_argument("--ckpt_path", type=str, default='checkpoints/base_512_v2/model.ckpt', help="checkpoint path")
    parser.add_argument("--config", type=str, default='configs/inference_t2v_512_v2.0.yaml', help="config (yaml) path")
    parser.add_argument("--prompt_file", type=str, default="prompts/videocomp_prompt.txt", help="a text file containing many prompts")
    parser.add_argument("--savedir", type=str, default="results/base_512_v2", help="results saving path")
    parser.add_argument("--savefps", type=str, default=10, help="video fps to generate")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt",)
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM",)
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)",)
    parser.add_argument("--bs", type=int, default=1, help="batch size for inference")
    parser.add_argument("--height", type=int, default=320, help="image height, in pixel space")
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--frames", type=int, default=-1, help="frames num to inference")
    parser.add_argument("--fps", type=int, default=28)
    parser.add_argument("--unconditional_guidance_scale", type=float, default=12.0, help="prompt classifier-free guidance")
    parser.add_argument("--unconditional_guidance_scale_temporal", type=float, default=None, help="temporal consistency guidance")
    ## for conditional i2v only
    parser.add_argument("--cond_input", type=str, default=None, help="data dir of conditional input")
    return parser

def process_bg(bg_list):
    bg = []
    for i in range(len(bg_list)):
        # 读取和处理背景图像
        bg_img = cv2.imread(bg_list[i])
        bg_img = cv2.resize(bg_img,(512,512))/255.0
        bg.append(bg_img)

def run_inference(args, gpu_num, gpu_no, **kwargs):
    ## step 1: model config
    ## -----------------------------------------------------------------
    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())
    model = instantiate_from_config(model_config)
    model = model.cuda(gpu_no)
    assert os.path.exists(args.ckpt_path), f"Error: checkpoint [{args.ckpt_path}] Not Found!"
    model = load_model_checkpoint(model, args.ckpt_path)
    model.eval()

    

    ## sample shape
    assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
    ## latent noise shape
    h, w = args.height // 8, args.width // 8
    frames = model.temporal_length if args.frames < 0 else args.frames
    channels = model.channels
    
    ## saving folders
    os.makedirs(args.savedir, exist_ok=True)

    ## step 2: load data
    ## -----------------------------------------------------------------
    # prompt文件
    assert os.path.exists(args.prompt_file), "Error: prompt file NOT Found!"
    prompt_list = load_prompts(args.prompt_file)
    num_samples = len(prompt_list)
    filename_list = [f"{id+1:04d}" for id in range(num_samples)]

    samples_split = num_samples // gpu_num
    residual_tail = num_samples % gpu_num
    print(f'[rank:{gpu_no}] {samples_split}/{num_samples} samples loaded.')
    indices = list(range(samples_split*gpu_no, samples_split*(gpu_no+1)))
    if gpu_no == 0 and residual_tail != 0:
        indices = indices + list(range(num_samples-residual_tail, num_samples))
    prompt_list_rank = [prompt_list[i] for i in indices]

    filename_list_rank = [filename_list[i] for i in indices]

    #bg视频文件
    assert os.path.exists(args.prompt_file), "Error: bg file NOT Found!"
    bg_list = [f.path for f in os.scandir(args.bg_path) if f.path.endswith('.mp4')]
    # bg_list = sorted(bg_list)
    video_batch = load_video_batch(filepath_list=bg_list, frame_stride=1, video_size=(320,512)).cuda()
    filename_list_rank = [i.split('/')[-1].split('.')[0] for i in bg_list]

    assert len(video_batch) == num_samples, f"Error: bg conditional input ({len(video_batch)}) NOT match prompt ({num_samples})!"

    ## step 3: sampler initialize and noise initialization
    ## -----------------------------------------------------------------
    ddim_sampler = DDIMSampler(model)
    ddim_sampler.make_schedule(ddim_num_steps=args.ddim_steps, ddim_eta=args.ddim_eta)

    video_latent_batch = []
    noise_video_latent_batch = []
    for i in range(num_samples):
        ## encode video as initial point
        video_latent = model.encode_first_stage_2DAE(video_batch[i].unsqueeze(0))
        t_enc = int(0.7*args.ddim_steps)
        noise_video_latent = ddim_sampler.stochastic_encode(video_latent, torch.tensor([t_enc-1]*args.bs).cuda())

        video_latent_batch.append(video_latent)
        noise_video_latent_batch.append(noise_video_latent)




    ## step 3: run over samples
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
        # filenames = bg_list[idx].split('/')[-1]
        noise_shape = [batch_size, channels, frames, h, w]
        fps = torch.tensor([args.fps]*batch_size).to(model.device).long()

        prompts = prompt_list_rank[idx_s:idx_e]
        if isinstance(prompts, str):
            prompts = [prompts]
        #prompts = batch_size * [""]
        text_emb = model.get_learned_conditioning(prompts)

        if args.mode == 'base':
            cond = {"c_crossattn": [text_emb], "fps": fps} #文本条件
        else:
            raise NotImplementedError

        #自己加的初始噪声
        # x_T = torch.randn((1,4,16,40,64), device=model.betas.device)
        # x_T = torch.zeros((1,4,16,40,64), device=model.betas.device)
        x_T = noise_video_latent_batch[idx]
        # x0 = video_latent_batch[idx]

        ## inference 没有传入初始noise 是在sampler中randn的
        # batch_samples = batch_ddim_sampling(model, cond, noise_shape, args.n_samples, \
        #                                         args.ddim_steps, args.ddim_eta, args.unconditional_guidance_scale, x_T=x_T, **kwargs)
        batch_variants = []
        for _ in range(args.n_samples):
            if ddim_sampler is not None:
                kwargs.update({"clean_cond": True})
                # samples, _ = ddim_sampler.sample(S=args.ddim_steps,
                #                                     conditioning=cond,
                #                                     batch_size=noise_shape[0],
                #                                     shape=noise_shape[1:],
                #                                     verbose=False,
                #                                     unconditional_guidance_scale=args.unconditional_guidance_scale,
                #                                     unconditional_conditioning=None,
                #                                     eta=args.ddim_eta,
                #                                     temporal_length=noise_shape[2],
                #                                     conditional_guidance_scale_temporal=None,
                #                                    # x0 = x0,
                #                                     x_T=x_T, 
                #                                     **kwargs
                #                                     )
                samples = ddim_sampler.decode(x_latent = x_T, 
                                              cond = cond, 
                                              t_start=t_enc, 
                                              unconditional_guidance_scale=args.unconditional_guidance_scale,
                                              unconditional_conditioning=None)
            ## reconstruct from latent to pixel space
            batch_images = model.decode_first_stage_2DAE(samples)
            batch_variants.append(batch_images)
        batch_variants = torch.stack(batch_variants, dim=1)
        batch_samples = batch_variants

        ## b,samples,c,t,h,w
        save_videos(batch_samples, args.savedir, filenames, fps=args.savefps) #把生成的视频保存下来

    print(f"Saved in {args.savedir}. Time used: {(time.time() - start):.2f} seconds")

if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("@CoLVDM Inference: %s"%now)
    parser = get_parser()
    args = parser.parse_args()
    seed_everything(args.seed)
    rank, gpu_num = 0, 1
    run_inference(args, gpu_num, rank)