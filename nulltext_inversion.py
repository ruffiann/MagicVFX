import torch
from torch.optim.adam import Adam
from tqdm.notebook import tqdm
import torch.nn.functional as nnf
import numpy as np
from typing import Union
import copy

#timestep_list = timesteps[0:total_steps-i]
@torch.no_grad()
def ddim_invert(model, alphas_cumprod, start_latent, cond, unconditional_conditioning=None,  unconditional_guidance_scale=1., timestep_list=None):
    latent = start_latent.clone().detach()
    b = latent.shape[0]
    intermediate_latents = []
    for i, step in enumerate(timestep_list):
        ts = torch.full((b,), step, device=model.betas.device, dtype=torch.long)
        e_t = model.apply_model(latent, ts, cond)
        e_t_uncond = model.apply_model(latent, ts, unconditional_conditioning)

        if unconditional_guidance_scale!=1.:
            noise_pred = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        current_t = max(0, ts.item() - (1000//50))
        next_t = ts.item()
        alpha_t = alphas_cumprod[current_t]
        alpha_t_next = alphas_cumprod[next_t]

        latent = (latent - (1-alpha_t).sqrt()*noise_pred)*(alpha_t_next.sqrt()/alpha_t.sqrt()) + (1-alpha_t_next).sqrt()*noise_pred

        intermediate_latents.append(latent)
    
    return latent
    



class NullInversion:
    def __init__(self, model, sampler, NUM_DDIM_STEPS=50, GUIDANCE_SCALE=7.5) -> None:
        self.model = model
        self.sampler = sampler
        self.NUM_DDIM_STEPS = NUM_DDIM_STEPS
        self.GUIDANCE_SCALE = GUIDANCE_SCALE

    def null_optimization(self, latents, uncond_embeddings, cond_embeddings, num_inner_steps=10, epsilon=1e-5): #latent是用ddim从image到噪声的invert
        uncond_embeddings_list = []
        latent_cur = latents[-1] #先翻过来从step1000的latent开始
        # bar = tqdm(total=num_inner_steps * self.NUM_DDIM_STEPS)

        b = latent_cur.shape[0]
        timesteps = self.sampler.ddim_timesteps
        time_range = np.flip(timesteps)
        for i in range(self.NUM_DDIM_STEPS):
            uncond_embeddings['c_crossattn'][0] = uncond_embeddings['c_crossattn'][0].clone().detach().requires_grad_(True)
            # uncond_embeddings.requires_grad = True

            # uncond_embeddings_c = uncond_embeddings['c_crossattn'][0].clone().detach().requires_grad_(True)
            # uncond_embeddings_fps = uncond_embeddings['fps']
            # uncond_embeddings = {'c_crossattn': [uncond_embeddings_c], 'fps': uncond_embeddings_fps}
            
            optimizer = Adam([uncond_embeddings['c_crossattn'][0]], lr=1e-2 * (1. - i / 100.))
            # optimizer = Adam([uncond_embeddings_c], lr=1e-2 * (1. - i / 100.))

            latent_prev = latents[len(latents) - i - 2]
            ts = torch.full((b,), time_range[i], device=self.model.betas.device, dtype=torch.long)

            with torch.no_grad():
                noise_pred_cond = self.sampler.p_sample_ddim(latent_cur, cond_embeddings, ts, ddim_inverse=True)

            for j in range(num_inner_steps):
                noise_pred_uncond = self.sampler.p_sample_ddim(latent_cur, uncond_embeddings, ts, ddim_inverse=True, req_grad=True)
                noise_pred = noise_pred_uncond + self.GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, ts, latent_cur)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                # loss.requires_grad_(True)
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                # bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            # for j in range(j + 1, num_inner_steps):
            #     bar.update()
            uncond_embeddings_list.append(copy.deepcopy(uncond_embeddings))
            with torch.no_grad():
                # context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, ts, False, cond_embeddings, uncond_embeddings)
        # bar.close()
        return uncond_embeddings_list
    
    def get_noise_pred_single(self, latent, t, cond):
        noise_pred = self.model.apply_model(latent, t, cond)
        # noise_pred = self.sampler.p_sample_ddim
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, cond_context=None, uncond_context=None):
        # latents_input = torch.cat([latents] * 2)
        # if context is None:
        #     context = self.context
        guidance_scale = 1 if is_forward else self.GUIDANCE_SCALE
        noise_pred_uncond = self.sampler.p_sample_ddim(latents, uncond_context, t, ddim_inverse=True)
        noise_prediction_text = self.sampler.p_sample_ddim(latents, cond_context, t, ddim_inverse=True)
        # noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents
    
    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = int(timestep) - self.sampler.ddpm_num_timesteps// self.NUM_DDIM_STEPS
        alpha_prod_t = self.sampler.alphas_cumprod[int(timestep)]
        alpha_prod_t_prev = self.sampler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.sampler.alphas_cumprod[0]
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample
    
    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        # timestep, next_timestep = min(int(timestep) - self.sampler.ddpm_num_timesteps// self.NUM_DDIM_STEPS, 999), int(timestep)
        timestep, next_timestep = min(int(timestep) + self.sampler.ddpm_num_timesteps// self.NUM_DDIM_STEPS, 999), int(timestep)
        alpha_prod_t = self.sampler.alphas_cumprod[timestep] if timestep >= 0 else self.sampler.alphas_cumprod[0]
        alpha_prod_t_next = self.sampler.alphas_cumprod[next_timestep] #self.sampler.alphas_cumprod.shape=1000
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def ddim_inversion(self, latent, cond):
        b = latent.shape[0]
        all_latent = [latent]
        latent = latent.clone().detach()
        timesteps = self.sampler.ddim_timesteps
        # time_range = np.flip(timesteps)
        # iterator = time_range
        for i, step in enumerate(timesteps):
            ts = torch.full((b,), step, device=self.model.betas.device, dtype=torch.long) #ts=tensor([981])
            noise_pred = self.sampler.p_sample_ddim(latent, cond, ts, ddim_inverse=True) #tensor 1*4*frames*40*64
            latent = self.next_step(noise_pred, ts, latent)
            all_latent.append(latent)
            del noise_pred
            torch.cuda.empty_cache()
        
        return all_latent #从step1到step1000的latent
    
    @torch.no_grad()
    def q_sample_inversion(self, latent, cond):
        b = latent.shape[0]
        all_latent = [latent]
        latent = latent.clone().detach()
        timesteps = self.sampler.ddim_timesteps
        for i, step in enumerate(timesteps):
            ts = torch.full((b,), step, device=self.model.betas.device, dtype=torch.long)
            latent = self.model.q_sample(latent, ts)
            all_latent.append(latent)
        
        return all_latent
    
def get_optimized_unconditional_embedding(model, sampler, x0_latent, cond, ddim_steps, cfg_scale): 
    '''
    x0是输入视频的未加噪latent
    cond中除了c_embeddings还有FPS
    '''
    Nullinversion = NullInversion(model, sampler, ddim_steps, cfg_scale)

    uc_embeddings = model.get_learned_conditioning([""])
    if isinstance(cond, dict):
        uncond_embeddings = {key:cond[key] for key in cond.keys()}
        uncond_embeddings.update({'c_crossattn': [uc_embeddings]})
    else:
        uncond_embeddings = uc_embeddings
    
    latents = Nullinversion.ddim_inversion(x0_latent, cond)
    # latents = Nullinversion.q_sample_inversion(x0_latent, cond)
    uncond_embeddings_list = Nullinversion.null_optimization(latents, uncond_embeddings, cond)

    return uncond_embeddings_list
    