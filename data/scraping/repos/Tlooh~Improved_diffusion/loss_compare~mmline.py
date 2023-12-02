
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import torch
from torch.nn import functional as F
import numpy as np
from PIL import Image
from diffusers.utils import  logging

from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline

from utils.obj_loss import loss1, loss2, loss3, loss4

from utils.ptp_utils import AttentionStore,aggregate_attention,view_L_images,text_under_L_image, text_under_image, view_images, view_L_images

from utils.align_utils import ObjectSplit, AlignEmbeds

logger = logging.get_logger(__name__)

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
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


class MyPipeline(StableDiffusionPipeline):
   
    _optional_components = ["safety_checker", "feature_extractor"]

    def _view_L_attention_per_index(self, attention_maps: torch.Tensor, tokens) -> List[torch.Tensor]:
        attention_map_per_token = []
        for idx in range(len(attention_maps)):
            image = attention_maps[idx]
            image = 255*image / image.max()
            image = image.unsqueeze(0)
            # image = image.numpy().astype(np.uint8)
            image = F.interpolate(image.unsqueeze(0), scale_factor = 16, mode ="nearest")[0].cpu().detach().numpy()  #shape : [1,256,256]
            image = image[0].astype(np.uint8)
            image = np.where(image>30, image, 0)
            image = text_under_L_image(image,self.tokenizer.decode(int(tokens[idx+1])))

            attention_map_per_token.append(image)
        
        images = view_L_images(attention_map_per_token)
        
    def _view_attention_per_index(self, attention_maps: torch.Tensor, tokens) -> List[torch.Tensor]:
        attention_map_per_token = []
        for idx in range(len(attention_maps)):
            image = attention_maps[idx]
            image = 255*image / image.max()
            image = image.unsqueeze(-1).expand(*image.shape,3)
            image = image.cpu().detach().numpy().astype(np.uint8)
            image = np.array(Image.fromarray(image).resize((256,256)))
            image = text_under_image(image,self.tokenizer.decode(int(tokens[idx+1])))

            attention_map_per_token.append(image)
        
        view_images(np.stack(attention_map_per_token, axis=0))


    def _aggregate_and_get_attention_per_token(self, attention_store: AttentionStore,  attention_res: int = 16, show_attention_per_step: bool = False):
        
        """ Aggregates the attention for each token and computes the max activation value for each token to alter. """
        attention_maps = aggregate_attention(
                attention_store=attention_store,
                res=attention_res,
                from_where=("up", "down", "mid"),
                is_cross=True,
                select=0)  #shape: [16,16,77]
        
        # 1: caculate the num of tokes(prompt)
        prompt = self.prompt
             
        num_tokens = len(self.tokenizer(prompt)['input_ids'])
        
        # 2: get the attention map of per toke(prompt)
        visiual_maps = []
        tokens = self.tokenizer.encode(prompt)
        for i in range(1, num_tokens-1):
            visiual_maps.append(attention_maps[:, :, i])
                   
        if show_attention_per_step:
            self._view_L_attention_per_index(visiual_maps, tokens)

        # 3. return the specific token's attention map 16x16
        attention_maps_tokens = [] 

        # 返回remove <begin>， <end>后 token对应的 attention map
        for idx in range(1, num_tokens - 1):
            single_token_attention_map  = attention_maps[:, :, idx]
            # single_token_attention_map *= 100
            # single_token_attention_map = F.softmax(single_token_attention_map, dim=-1)

            attention_maps_tokens.append(single_token_attention_map)
        
        
        return attention_maps_tokens



    @staticmethod
    def _compute_loss(
        self,
        attention_maps_tokens: List[torch.Tensor], 
        indices_to_compute=None) -> torch.Tensor:
        """ Computes the IOU loss using the maximum attention value for each token. eg:传入 indices_to_compute = [(1,3), (2,5)]
        """
        tokens = self.tokenizer(self.prompt)['input_ids'][1:-1]
        
        
        losses = 0
        if indices_to_compute is not None:

            for indices in indices_to_compute:

                obj_token1 = int(tokens[indices[0]]) 
                obj_token2 = int(tokens[indices[1]])
                print(f"compute loss between {self.tokenizer.decode(obj_token1)} and {self.tokenizer.decode(obj_token2)}.")

                loss = loss4(attention_maps_tokens[indices[0]],attention_maps_tokens[indices[1]])


                losses += loss
        
        else:
            raise ValueError("请传入 indices_to_computer")
        
        return losses
   
    
    
    @staticmethod
    def _update_latent(latents: torch.Tensor, loss: torch.Tensor, step_size: float) -> torch.Tensor:
        """ Update the latent according to the computed loss. """
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
        # grad_cond shape :[1,4,64,64]
        latents = latents - step_size * grad_cond
        return latents
         
    

    def _perform_iterative_refinement_step(self,
                                           latents: torch.Tensor,
                                           loss: torch.Tensor,
                                           threshold: float,
                                           text_embeddings: torch.Tensor,
                                           text_input,
                                           attention_store: AttentionStore,
                                           step_size: float,
                                           t: int,
                                           attention_res: int = 16,
                                           max_refinement_steps: int = 20,
                                           show_attention_per_step: bool = False):
        
        iteration = 0
        target_loss = threshold
        while loss > target_loss:
            iteration += 1

            latents = latents.clone().detach().requires_grad_(True)
            noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample
            self.unet.zero_grad()

            attention_maps_tokens = self._aggregate_and_get_attention_per_token(
                        attention_store = attention_store,
                        attention_res=attention_res,
                        show_attention_per_step = show_attention_per_step)
            
            loss = self._compute_loss(attention_maps_tokens)

            if loss != 0:
                latents = self._update_latent(latents, loss, step_size)
            
            with torch.no_grad():
                noise_pred_uncond = self.unet(latents, t, encoder_hidden_states=text_embeddings[0].unsqueeze(0)).sample

                noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample
            
            print(f'\t Try {iteration}. | target loss: {target_loss} | Loss : {loss}')

            if iteration >= max_refinement_steps:
                print(f'\t Exceeded max number of iterations ({max_refinement_steps})!' )

                break
        
        # Run one more time but don't compute gradients and update the latents.
        # We just need to compute the new loss - the grad update will occur below
        latents = latents.clone().detach().requires_grad_(True)
        noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample
        self.unet.zero_grad()

        attention_maps_tokens = self._aggregate_and_get_attention_per_token(
                        attention_store = attention_store,
                        attention_res=attention_res,
                        show_attention_per_step = show_attention_per_step)
            
        loss = self._compute_loss(attention_maps_tokens)

        print(f"\t Finished with loss of: {loss}")

        return loss, latents, attention_maps_tokens


            

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        attention_store: AttentionStore,
        attention_res: int = 16,
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
        max_iter_to_alter: Optional[int] = 25,
        run_standard_sd: bool = False,
        run_structure_sd: bool = False,
        thresholds: Optional[dict] = {10:1.2, 20: 1 , 30:0.95, 40:0.9},
        scale_factor: int = 20,
        scale_range: Tuple[float, float] = (1., 0.5),
        show_attention_per_step: bool = False,
        iterative_refinement: bool = True,
        indices_to_compute = None,
        pos_to_replace = [(0, 11), (7,11)]
    ):
        

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        self.prompt = prompt
        
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )


        if not run_structure_sd:

            prompt_embeds = self._encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=text_encoder_lora_scale,
            )


        else:
            split_object = ObjectSplit(prompt)
            obj_info, words, noun_chunk = split_object()
            self.nps = list(obj_info.keys())
            self.spans = list(obj_info.values())

            struct_prompt_process = AlignEmbeds(
                obj_info=obj_info,
                words = words,
                tokenizer=self.tokenizer,
                text_encoder=self.text_encoder,
                device=self.device)
            
            prompt_embeds = struct_prompt_process(pos_to_replace = pos_to_replace)
            
       
        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        scale_range = np.linspace(scale_range[0], scale_range[1], len(self.scheduler.timesteps))

        if max_iter_to_alter is None:
            max_iter_to_alter = len(self.scheduler.timesteps) + 1

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                with torch.enable_grad():

                    latents = latents.clone().detach().requires_grad_(True)
                    # Forward pass of denoising with text conditioning
                    noise_pred_text = self.unet(latents, t,
                                                encoder_hidden_states=prompt_embeds[1].unsqueeze(0), cross_attention_kwargs=cross_attention_kwargs).sample
                
                #     self.unet.zero_grad()
                    # del noise_pred_text
                    # torch.cuda.empty_cache()

                    attention_maps_tokens = self._aggregate_and_get_attention_per_token(
                        attention_store = attention_store,
                        attention_res=attention_res,
                        show_attention_per_step = show_attention_per_step)

                    if not run_standard_sd:

                        # loss = self._compute_loss(
                        #     self,
                        #     attention_maps_tokens=attention_maps_tokens,
                        #     indices_to_compute=indices_to_compute)

                        # 判断有无达到阈值，是否迭代去噪
                        if iterative_refinement and i in thresholds.keys() and loss > thresholds[i]:
                            del noise_pred_text
                            torch.cuda.empty_cache()
                            loss, latents, attention_maps_tokens = self._perform_iterative_refinement_step(
                                latents=latents,
                                loss=loss,
                                threshold=thresholds[i],
                                text_embeddings=prompt_embeds,
                                text_input=None,
                                attention_store=attention_store,
                                step_size=scale_factor * np.sqrt(scale_range[i]),
                                t=t,
                                max_refinement_steps = 20,
                                attention_res=attention_res,
                                show_attention_per_step = False)
                            

                        if i < max_iter_to_alter :
                        # caculate loss
                            loss = self._compute_loss(
                                self,
                                attention_maps_tokens,
                                indices_to_compute=indices_to_compute)
                            
                            if loss != 0:
                                # use loss to update latents
                                latents = self._update_latent(latents=latents, loss=loss, step_size=scale_factor * np.sqrt(scale_range[i]))
                            print(f'Iteration {i} | Loss: {loss:0.4f}')

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)
              
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)