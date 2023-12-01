import os
from typing import Any, Callable, Dict, List, Optional, Union
import torch
import torch.nn.functional as F
import numpy as np
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler, Transformer2DModel
from PIL import Image
import matplotlib.pyplot as plt

def find_subsequence_indices_np(larger, smaller):
    window_view = np.lib.stride_tricks.sliding_window_view(larger, len(smaller))
    return np.where(np.all(window_view == smaller, axis=1))[0]

def get_token_indicies(tokenizer, prompt, subseq):
    if type(prompt) == str:
        prompt = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids.numpy()[0]
    if type(subseq) == str:
        subseq = tokenizer(subseq, padding="longest", return_tensors="pt").input_ids.numpy()[0, 1:-1]
    idx = []
    for window_start in find_subsequence_indices_np(prompt, subseq):
        idx.extend(list(range(window_start, window_start + len(subseq))))
    return idx

def normalize_tensor(x):
    x_min = x.min()
    x_range = x.max() - x.min()
    return (x - x_min) / x_range

respect_to = ''

class AttentionMapExtractor():

    attention_maps = []
    def __init__(self, attentions, map_height=64, map_width=64, size_hints=[], auto_reset=True):
        self.map_height = map_height
        self.map_width = map_width
        self.auto_reset = auto_reset
        self.hooks = []
        self.size_hints = size_hints
        for attention in attentions:
            self.hooks.append(attention.register_forward_pre_hook(self.hook_fn, with_kwargs=True))

    def compute_attention(self, Q, K, V=None, attn_mask=None, dropout_p=0.0, is_causal=False): #From https://pytorch.org/docs/2.0/generated/torch.nn.functional.scaled_dot_product_attention.html
        L, S = Q.shape[-2], K.shape[-2]
        if attn_mask is None:
            attn_weight = torch.softmax(Q @ K.transpose(-2, -1) / np.sqrt(Q.shape[-1]), dim=-1)
        else:
            attn_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0) if is_causal else attn_mask
            attn_mask = attn_mask.masked_fill(not attn_mask, -float('inf')) if attn_mask.dtype==torch.bool else attn_mask
            attn_weight = torch.softmax((Q @ K.transpose(-2, -1) / np.sqrt(Q.shape[-1])) + attn_mask, dim=-1)
        attn_weight = F.dropout(attn_weight, dropout_p)
        if V is None:
            return attn_weight
        return attn_weight @ V
    
    def process_attention_maps(self, token_indices, positive_prompt=True, binarize_s=10):
        #2, attention heads, h, w, sequence length
        #torch.Size([2, 8, 64, 64, 77])
        batch = 1 if positive_prompt else 0
        attention_maps_rescaled = []
        for attention_map in self.attention_maps:
            rescaled = attention_map[batch:batch+1, :, :, :, token_indices].permute(0, 1, 4, 2, 3).flatten(1, 2)
            rescaled = F.interpolate(rescaled, (self.map_height, self.map_width), mode='bicubic')
            attention_maps_rescaled.append(rescaled)

        all_maps = torch.cat(attention_maps_rescaled, dim=1)
        raw_map = torch.mean(all_maps, dim=1)
        if binarize_s <= 0:
            return raw_map
        binarized_maps = []
        for batch in raw_map:
            binarized_maps.append(normalize_tensor(F.sigmoid(binarize_s * (normalize_tensor(batch) - 0.5))))
        return torch.stack(binarized_maps, dim=0)


    #self, module, input, output
    def hook_fn(self, attn, hidden_states, kwargs):
        if self.auto_reset:
            if (len(self.attention_maps) > 0) and (len(self.attention_maps) % len(self.hooks) == 0):
                self.attention_maps.clear()

        if(type(hidden_states) == tuple):
            hidden_states = hidden_states[0]

        encoder_hidden_states = kwargs['encoder_hidden_states']
        sequence_length = encoder_hidden_states.shape[1]
        if attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        attention_mask = kwargs.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        batch_size, _, npix = hidden_states.shape 
        head_dim = npix // attn.heads

        query = attn.to_q(hidden_states).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = attn.to_k(encoder_hidden_states).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        attention_map = self.compute_attention(query, key, attn_mask=attention_mask)

        map_idx = len(self.attention_maps) % len(self.hooks)
        if map_idx >= 0 and map_idx < len(self.size_hints):
            local_h, local_w = self.size_hints[map_idx]
        else:
            local_h = int(np.sqrt(attention_map.shape[-2]))
            local_w = attention_map.shape[-2] // local_h
            if (local_h * local_w) != attention_map.shape[-2]:
                local_h = attention_map.shape[-2]
                local_w = 1
        
        attention_map = attention_map.view(*attention_map.shape[:-2], local_h, local_w, attention_map.shape[-1])

        self.attention_maps.append(attention_map)

    def __del__(self):
        for hook in self.hooks:
            hook.remove()

class StableDiffusionPipelineCustom():
    def __init__(self, sdp):
        self.sdp = sdp
    def __call__(self, *args, **kwargs):
        return self.custom_pipeline(self.sdp, *args, **kwargs)

    def rescale_noise_cfg(self, noise_cfg, noise_pred_text, guidance_rescale=0.0):
        """
        Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
        Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
        """
        std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
        std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
        # rescale the results from guidance (fixes overexposure)
        noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
        # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
        noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
        return noise_cfg

    #torch.no_grad()
    def custom_pipeline(
        self, sdp,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            guidance_rescale (`float`, *optional*, defaults to 0.7):
                Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
                [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
                Guidance rescale factor should fix overexposure when using zero terminal SNR.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        # 0. Default height and width to unet
        height = height or sdp.unet.config.sample_size * sdp.vae_scale_factor
        width = width or sdp.unet.config.sample_size * sdp.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        sdp.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = sdp._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )

        prompt_embeds = sdp._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 4. Prepare timesteps
        sdp.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = sdp.scheduler.timesteps

        #(Extra step) attention hooks
        sigmas = sdp.scheduler.add_noise(torch.zeros(1,), torch.ones(1,), timesteps).tolist()
        attention_blocks = sdp.unet.down_blocks + [sdp.unet.mid_block] + sdp.unet.up_blocks
        attentions = ([], [])
        #A questionable fix for questionable code
        sizes_precomputed = [(64, 64), (32, 32), (16, 16), (8, 8), (8, 8), (8, 8), (16, 16), (32, 32), (64, 64)]
        for attention_block, map_size in zip(attention_blocks, sizes_precomputed):
            if not hasattr(attention_block, 'attentions'):
                continue
            for transformer2d in attention_block.attentions:
                if type(transformer2d) != Transformer2DModel:
                    continue
                for basicTransformerBlock in transformer2d.transformer_blocks:
                    if basicTransformerBlock.only_cross_attention:
                        attentions[0].append(basicTransformerBlock.attn1)
                        attentions[1].append(map_size)
                    attentions[0].append(basicTransformerBlock.attn2)
                    attentions[1].append(map_size)

        at_map_extractor = AttentionMapExtractor(attentions[0], map_height=64, map_width=64, size_hints=attentions[1])
        start_steps_with_guidance = int((0.1875 * num_inference_steps) + 0.5)
        end_steps_no_guidance = num_inference_steps - (num_inference_steps // 32)
        use_guidance_steps = list(range(0, start_steps_with_guidance, 1)) + list(range(start_steps_with_guidance, end_steps_no_guidance, 2))
        save_steps = list(range(0, num_inference_steps-1, 5)) + [num_inference_steps - 1]
        token_indicies = get_token_indicies(sdp.tokenizer, prompt, respect_to)
        target_attention_map = torch.zeros((64, 64), device=device, dtype=torch.float16)
        target_attention_map[:32, :32] = 1.0

        # 5. Prepare latent variables
        num_channels_latents = sdp.unet.config.in_channels
        latents = sdp.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        ).requires_grad_()

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = sdp.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * sdp.scheduler.order
        with sdp.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                if i in use_guidance_steps:
                    latents.requires_grad_()
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = sdp.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = sdp.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0].detach()

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

                self_guidance = 0
                if i in use_guidance_steps and do_classifier_free_guidance:
                    self_guidance_scale = 1500 * sigmas[i]
                    attention_map = at_map_extractor.process_attention_maps(token_indicies)[0]
                    energy_function = F.l1_loss(attention_map, target_attention_map)
                    self_guidance = torch.autograd.grad(energy_function, latents)[0] * self_guidance_scale

                latents = latents.detach()

                with torch.no_grad():
                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                        noise_pred = noise_pred + self_guidance

                    if do_classifier_free_guidance and guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = self.rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = sdp.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                    if i in save_steps:
                        image = sdp.vae.decode(latents / sdp.vae.config.scaling_factor, return_dict=False)[0]
                        do_denormalize = [True] * image.shape[0]
                        image = sdp.image_processor.postprocess(image, do_denormalize=do_denormalize)
                        image[0].save(f'out/sd_denoising_{i}.png')
                        attention_map = at_map_extractor.process_attention_maps(token_indicies)[0] * 255.0
                        Image.fromarray(attention_map.detach().cpu().numpy().astype(np.uint8)).resize((512, 512)).save(f'out/sd_attention_{i}.png')
                        
                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % sdp.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            callback(i, t, latents)

        with torch.no_grad():
            if not output_type == "latent":
                image = sdp.vae.decode(latents / sdp.vae.config.scaling_factor, return_dict=False)[0]
                #image, has_nsfw_concept = sdp.run_safety_checker(image, device, prompt_embeds.dtype)
                has_nsfw_concept = None
            else:
                image = latents
                has_nsfw_concept = None

            if has_nsfw_concept is None:
                do_denormalize = [True] * image.shape[0]
            else:
                do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

            image = sdp.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload last model to CPU
        if hasattr(sdp, "final_offload_hook") and sdp.final_offload_hook is not None:
            sdp.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return {
            'images': image,
            'nsfw_content_detected': has_nsfw_concept,
        }

def run_model(sd_prompt, sd_prompt_negative):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    sd_pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
    ).to(device)
    #sd_pipe.scheduler = UniPCMultistepScheduler.from_config(sd_pipe.scheduler.config)

    sd_pipe_custom = StableDiffusionPipelineCustom(sd_pipe)
    generator = torch.Generator(device=device)
    generator.manual_seed(19081998)


    out = sd_pipe_custom(sd_prompt, generator=generator, num_inference_steps=50)

    plt.imshow(out['images'][0])
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--sd_prompt', type=str, default='A cat on the beach next to a beach ball')
    parser.add_argument('--sd_nprompt', type=str, default=None)
    parser.add_argument('--atten', type=str, default=None)
    args = parser.parse_args()
    if args.atten is None:
        respect_to = ' '.join(args.sd_prompt.split()[-2:])
    else:
        respect_to = args.atten
    os.makedirs('out', exist_ok=True)

    run_model(args.sd_prompt, args.sd_nprompt)