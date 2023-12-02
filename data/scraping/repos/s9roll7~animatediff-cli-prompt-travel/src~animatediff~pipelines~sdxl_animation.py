# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers import LCMScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import (FromSingleFileMixin, LoraLoaderMixin,
                               TextualInversionLoaderMixin)
from diffusers.models import AutoencoderKL, ControlNetModel
from diffusers.models.attention_processor import (AttnProcessor2_0,
                                                  LoRAAttnProcessor2_0,
                                                  LoRAXFormersAttnProcessor,
                                                  XFormersAttnProcessor)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (BaseOutput, is_accelerate_available,
                             is_accelerate_version, logging,
                             replace_example_docstring)
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from transformers import (CLIPTextModel, CLIPTextModelWithProjection,
                          CLIPTokenizer)

from animatediff.ip_adapter import IPAdapterPlusXL, IPAdapterXL
from animatediff.pipelines.animation import PromptEncoder, RegionMask
from animatediff.pipelines.context import (get_context_scheduler,
                                           get_total_steps)
from animatediff.sdxl_models.unet import UNet3DConditionModel
from animatediff.utils.control_net_lllite import ControlNetLLLite
from animatediff.utils.lpw_stable_diffusion_xl import \
    get_weighted_text_embeddings_sdxl2
from animatediff.utils.util import (get_tensor_interpolation_method, show_gpu,
                                    stopwatch_record, stopwatch_start,
                                    stopwatch_stop)


class PromptEncoderSDXL(PromptEncoder):
    def __init__(
            self,
            pipe,
            device,
            latents_device,
            num_videos_per_prompt,
            do_classifier_free_guidance,
            region_condi_list,
            negative_prompt,
            is_signle_prompt_mode,
            clip_skip,
            multi_uncond_mode
        ):
        self.pipe = pipe
        self.is_single_prompt_mode=is_signle_prompt_mode
        self.do_classifier_free_guidance = do_classifier_free_guidance

        uncond_num = 0
        if do_classifier_free_guidance:
            if multi_uncond_mode:
                uncond_num = len(region_condi_list)
            else:
                uncond_num = 1

        self.uncond_num = uncond_num

        ### text

        prompt_nums = []
        prompt_map_list = []
        prompt_list = []

        for condi in region_condi_list:
            _prompt_map = condi["prompt_map"]
            prompt_map_list.append(_prompt_map)
            _prompt_map = dict(sorted(_prompt_map.items()))
            _prompt_list = [_prompt_map[key_frame] for key_frame in _prompt_map.keys()]
            prompt_nums.append( len(_prompt_list) )
            prompt_list += _prompt_list

        (prompt_embeds_list, negative_prompt_embeds_list,
         pooled_prompt_embeds_list, negative_pooled_prompt_embeds_list) = get_weighted_text_embeddings_sdxl2(
             pipe, prompt_list, [negative_prompt], latents_device
        )

        self.prompt_embeds_dtype = prompt_embeds_list[0].dtype


        if do_classifier_free_guidance:
            negative = negative_prompt_embeds_list
            negative_pooled = negative_pooled_prompt_embeds_list
            positive = prompt_embeds_list
            positive_pooled = pooled_prompt_embeds_list
        else:
            positive = prompt_embeds_list
            positive_pooled = pooled_prompt_embeds_list

        if pipe.ip_adapter:
            pipe.ip_adapter.set_text_length(positive[0].shape[1])

        prompt_embeds_region_list = []
        pooled_embeds_region_list = []

        if do_classifier_free_guidance:
            prompt_embeds_region_list = [
                {
                    0:negative[0]
                }
            ] * uncond_num + prompt_embeds_region_list
            pooled_embeds_region_list = [
                {
                    0:negative_pooled[0]
                }
            ] * uncond_num + pooled_embeds_region_list

        pos_index = 0
        for prompt_map, num in zip(prompt_map_list, prompt_nums):
            prompt_embeds_map={}
            pooled_embeds_map={}
            pos = positive[pos_index:pos_index+num]
            pos_pooled = positive_pooled[pos_index:pos_index+num]

            for i, key_frame in enumerate(prompt_map):
                prompt_embeds_map[key_frame] = pos[i]
                pooled_embeds_map[key_frame] = pos_pooled[i]

            prompt_embeds_region_list.append( prompt_embeds_map )
            pooled_embeds_region_list.append( pooled_embeds_map )
            pos_index += num

        if do_classifier_free_guidance:
            prompt_map_list = [
                {
                    0:negative_prompt
                }
            ] * uncond_num + prompt_map_list

        self.prompt_map_list = prompt_map_list
        self.prompt_embeds_region_list = prompt_embeds_region_list
        self.pooled_embeds_region_list = pooled_embeds_region_list

        ### image
        if pipe.ip_adapter:

            ip_im_nums = []
            ip_im_map_list = []
            ip_im_list = []

            for condi in region_condi_list:
                _ip_im_map = condi["ip_adapter_map"]["images"]
                ip_im_map_list.append(_ip_im_map)
                _ip_im_map = dict(sorted(_ip_im_map.items()))
                _ip_im_list = [_ip_im_map[key_frame] for key_frame in _ip_im_map.keys()]
                ip_im_nums.append( len(_ip_im_list) )
                ip_im_list += _ip_im_list

            positive, negative = pipe.ip_adapter.get_image_embeds(ip_im_list)

            positive = positive.to(device=latents_device)
            negative = negative.to(device=latents_device)

            bs_embed, seq_len, _ = positive.shape
            positive = positive.repeat(1, 1, 1)
            positive = positive.view(bs_embed * 1, seq_len, -1)

            bs_embed, seq_len, _ = negative.shape
            negative = negative.repeat(1, 1, 1)
            negative = negative.view(bs_embed * 1, seq_len, -1)

            if do_classifier_free_guidance:
                negative = negative.chunk(negative.shape[0], 0)
                positive = positive.chunk(positive.shape[0], 0)
            else:
                positive = positive.chunk(positive.shape[0], 0)

            im_prompt_embeds_region_list = []

            if do_classifier_free_guidance:
                im_prompt_embeds_region_list = [
                    {
                        0:negative[0]
                    }
                ] * uncond_num + im_prompt_embeds_region_list

            pos_index = 0
            for ip_im_map, num in zip(ip_im_map_list, ip_im_nums):
                im_prompt_embeds_map={}
                pos = positive[pos_index:pos_index+num]

                for i, key_frame in enumerate(ip_im_map):
                    im_prompt_embeds_map[key_frame] = pos[i]

                im_prompt_embeds_region_list.append( im_prompt_embeds_map )
                pos_index += num


            if do_classifier_free_guidance:
                ip_im_map_list = [
                    {
                        0:None
                    }
                ] * uncond_num + ip_im_map_list


            self.ip_im_map_list = ip_im_map_list
            self.im_prompt_embeds_region_list = im_prompt_embeds_region_list

    def is_uncond_layer(self, layer_index):
        return self.uncond_num > layer_index


    def _get_current_prompt_embeds_from_text(
            self,
            prompt_map,
            prompt_embeds_map,
            pooled_embeds_map,
            center_frame = None,
            video_length : int = 0
            ):

        key_prev = list(prompt_map.keys())[-1]
        key_next = list(prompt_map.keys())[0]

        for p in prompt_map.keys():
            if p > center_frame:
                key_next = p
                break
            key_prev = p

        dist_prev = center_frame - key_prev
        if dist_prev < 0:
            dist_prev += video_length
        dist_next = key_next - center_frame
        if dist_next < 0:
            dist_next += video_length

        if key_prev == key_next or dist_prev + dist_next == 0:
            return prompt_embeds_map[key_prev], pooled_embeds_map[key_prev]

        rate = dist_prev / (dist_prev + dist_next)

        return (get_tensor_interpolation_method()( prompt_embeds_map[key_prev], prompt_embeds_map[key_next], rate ),
                get_tensor_interpolation_method()( pooled_embeds_map[key_prev], pooled_embeds_map[key_next], rate ))

    def get_current_prompt_embeds_from_text(
            self,
            center_frame = None,
            video_length : int = 0
            ):
        outputs = ()
        outputs2 = ()
        for prompt_map, prompt_embeds_map, pooled_embeds_map in zip(self.prompt_map_list, self.prompt_embeds_region_list, self.pooled_embeds_region_list):
            embs,embs2 = self._get_current_prompt_embeds_from_text(
                prompt_map,
                prompt_embeds_map,
                pooled_embeds_map,
                center_frame,
                video_length)
            outputs += (embs,)
            outputs2 += (embs2,)

        return outputs, outputs2

    def get_current_prompt_embeds_single(
            self,
            context: List[int] = None,
            video_length : int = 0
            ):
        center_frame = context[len(context)//2]
        text_emb, pooled_emb = self.get_current_prompt_embeds_from_text(center_frame, video_length)
        text_emb = torch.cat(text_emb)
        pooled_emb = torch.cat(pooled_emb)
        if self.pipe.ip_adapter:
            image_emb = self.get_current_prompt_embeds_from_image(center_frame, video_length)
            image_emb = torch.cat(image_emb)
            return torch.cat([text_emb,image_emb], dim=1), pooled_emb
        else:
            return text_emb, pooled_emb

    def get_current_prompt_embeds_multi(
            self,
            context: List[int] = None,
            video_length : int = 0
            ):

        emb_list = []
        pooled_emb_list = []
        for c in context:
            t,p = self.get_current_prompt_embeds_from_text(c, video_length)
            for i, (emb, pooled) in enumerate(zip(t,p)):
                if i >= len(emb_list):
                    emb_list.append([])
                    pooled_emb_list.append([])
                emb_list[i].append(emb)
                pooled_emb_list[i].append(pooled)

        text_emb = []
        for emb in emb_list:
            emb = torch.cat(emb)
            text_emb.append(emb)
        text_emb = torch.cat(text_emb)

        pooled_emb = []
        for emb in pooled_emb_list:
            emb = torch.cat(emb)
            pooled_emb.append(emb)
        pooled_emb = torch.cat(pooled_emb)

        if self.pipe.ip_adapter == None:
            return text_emb, pooled_emb

        emb_list = []
        for c in context:
            t = self.get_current_prompt_embeds_from_image(c, video_length)
            for i, emb in enumerate(t):
                if i >= len(emb_list):
                    emb_list.append([])
                emb_list[i].append(emb)

        image_emb = []
        for emb in emb_list:
            emb = torch.cat(emb)
            image_emb.append(emb)
        image_emb = torch.cat(image_emb)

        return torch.cat([text_emb,image_emb], dim=1), pooled_emb

    '''
    def get_current_prompt_embeds(
            self,
            context: List[int] = None,
            video_length : int = 0
            ):
        return self.get_current_prompt_embeds_single(context,video_length) if self.is_single_prompt_mode else self.get_current_prompt_embeds_multi(context,video_length)

    def get_prompt_embeds_dtype(self):
        return self.prompt_embeds_dtype

    def get_condi_size(self):
        return len(self.prompt_embeds_region_list)
    '''








@dataclass
class AnimatePipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    videos: Union[torch.Tensor, np.ndarray]



logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionXLPipeline

        >>> pipe = StableDiffusionXLPipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt).images[0]
        ```
"""


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
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


class AnimationPipeline(DiffusionPipeline, FromSingleFileMixin, LoraLoaderMixin, TextualInversionLoaderMixin):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion XL.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    In addition the pipeline inherits the following loading methods:
        - *LoRA*: [`StableDiffusionXLPipeline.load_lora_weights`]
        - *Ckpt*: [`loaders.FromSingleFileMixin.from_single_file`]

    as well as the following saving methods:
        - *LoRA*: [`loaders.StableDiffusionXLPipeline.save_lora_weights`]

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion XL uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_encoder_2 ([` CLIPTextModelWithProjection`]):
            Second frozen text-encoder. Stable Diffusion XL uses the text and pool portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the
            [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
            variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`CLIPTokenizer`):
            Second Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
    """
    model_cpu_offload_seq = "text_encoder->text_encoder_2->unet->vae"

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet3DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: Optional[bool] = None,
        controlnet_map: Dict[ str , ControlNetModel ]=None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
        )
        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.default_sample_size = self.unet.config.sample_size

        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )
        self.controlnet_map = controlnet_map


    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_slicing
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_slicing
    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_tiling
    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_tiling
    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def __enable_model_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        model_sequence = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )
        model_sequence.extend([self.unet, self.vae])

        hook = None
        for cpu_offloaded_model in model_sequence:
            _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)

        # We'll offload the last model manually.
        self.final_offload_hook = hook

    def encode_prompt(
        self,
        prompt: str,
        prompt_2: Optional[str] = None,
        device: Optional[torch.device] = None,
        num_videos_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            device: (`torch.device`):
                torch device
            num_videos_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            # textual inversion: procecss multi-vector tokens if necessary
            prompt_embeds_list = []
            prompts = [prompt, prompt_2]
            for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    prompt = self.maybe_convert_prompt(prompt, tokenizer)

                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                text_input_ids = text_inputs.input_ids
                untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
                ):
                    removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                    logger.warning(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {tokenizer.model_max_length} tokens: {removed_text}"
                    )

                prompt_embeds = text_encoder(
                    text_input_ids.to(device),
                    output_hidden_states=True,
                )

                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds.hidden_states[-2]

                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        # get unconditional embeddings for classifier free guidance
        zero_out_negative_prompt = negative_prompt is None and self.config.force_zeros_for_empty_prompt
        if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        elif do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt

            uncond_tokens: List[str]
            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt, negative_prompt_2]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = [negative_prompt, negative_prompt_2]

            negative_prompt_embeds_list = []
            for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    negative_prompt = self.maybe_convert_prompt(negative_prompt, tokenizer)

                max_length = prompt_embeds.shape[1]
                uncond_input = tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                negative_prompt_embeds = text_encoder(
                    uncond_input.input_ids.to(device),
                    output_hidden_states=True,
                )
                # We are only ALWAYS interested in the pooled output of the final text encoder
                negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

                negative_prompt_embeds_list.append(negative_prompt_embeds)

            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_videos_per_prompt).view(
            bs_embed * num_videos_per_prompt, -1
        )
        if do_classifier_free_guidance:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_videos_per_prompt).view(
                bs_embed * num_videos_per_prompt, -1
            )

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs


    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
        do_normalize=False,
    ):
        if do_normalize == False:
            image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        else:
            image = self.image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        #if do_classifier_free_guidance and not guess_mode:
        #    image = torch.cat([image] * 2)

        return image


    def check_inputs(
        self,
        prompt,
        prompt_2,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        negative_prompt_2=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        '''
        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
        '''
        if callback_steps is not None:
            if not isinstance(callback_steps, list):
                raise ValueError("`callback_steps` has to be a list of positive integers.")
            for callback_step in callback_steps:
                if not isinstance(callback_step, int) or callback_step <= 0:
                    raise ValueError("`callback_steps` has to be a list of positive integers.")

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. Make sure to generate `negative_pooled_prompt_embeds` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def __prepare_latents(self, batch_size, single_model_length, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, single_model_length, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        video_length,
        height,
        width,
        dtype,
        device,
        generator,
        img2img_map,
        timestep,
        latents=None,
        is_strength_max=True,
        return_noise=True,
        return_image_latents=True,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # Offload text encoder if `enable_model_cpu_offload` was enabled
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.text_encoder_2.to("cpu")
            torch.cuda.empty_cache()


        image_latents = None

        if img2img_map:

            image_latents = torch.zeros(shape, device=device, dtype=dtype)
            for frame_no in img2img_map["images"]:
                img = img2img_map["images"][frame_no]
                img = self.image_processor.preprocess(img)
                img = img.to(device="cuda", dtype=self.vae.dtype)
                img = self.vae.encode(img).latent_dist.sample(generator)
                img = self.vae.config.scaling_factor * img
                img = torch.cat([img], dim=0)
                image_latents[:,:,frame_no,:,:] = img.to(device=device, dtype=dtype)

        else:
            is_strength_max = True


        if latents is None:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = noise if is_strength_max else self.scheduler.add_noise(image_latents, noise, timestep)
            latents = latents * self.scheduler.init_noise_sigma if is_strength_max else latents
        else:
            noise = latents.to(device)
            latents = noise * self.scheduler.init_noise_sigma


        outputs = (latents.to(device, dtype),)

        if return_noise:
            outputs += (noise.to(device, dtype),)

        if return_image_latents:
            if image_latents is not None:
                outputs += (image_latents.to(device, dtype),)
            else:
                outputs += (None,)

        return outputs


    def __prepare_latents(
        self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None, add_noise=True
    ):

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt

        if image.shape[1] == 4:
            init_latents = image

        else:
            # make sure the VAE is in float32 mode, as it overflows in float16
            if self.vae.config.force_upcast:
                image = image.float()
                self.vae.to(dtype=torch.float32)

            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            elif isinstance(generator, list):
                init_latents = [
                    self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = self.vae.encode(image).latent_dist.sample(generator)

            if self.vae.config.force_upcast:
                self.vae.to(dtype)

            init_latents = init_latents.to(dtype)
            init_latents = self.vae.config.scaling_factor * init_latents

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        if add_noise:
            shape = init_latents.shape
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # get latents
            init_latents = self.scheduler.add_noise(init_latents, noise, timestep)

        latents = init_latents

        return latents






    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + self.text_encoder_2.config.projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale.StableDiffusionUpscalePipeline.upcast_vae
    def upcast_vae(self):
        dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                LoRAXFormersAttnProcessor,
                LoRAAttnProcessor2_0,
            ),
        )
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)


    def decode_latents(self, latents: torch.Tensor):
        video_length = latents.shape[2]
        latents = 1 / self.vae.config.scaling_factor * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # video = self.vae.decode(latents).sample
        video = []
        for frame_idx in range(latents.shape[0]):
            video.append(
#                self.vae.decode(latents[frame_idx : frame_idx + 1].to(self.vae.device, self.vae.dtype)).sample.cpu()
                self.vae.decode(latents[frame_idx : frame_idx + 1].to("cuda", self.vae.dtype)).sample.cpu()
            )
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.float().numpy()
        return video

    def get_img2img_timesteps(self, num_inference_steps, strength, device):
        strength = min(1, max(0,strength))
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start


    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        single_model_length: Optional[int] = 16,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[List[int]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,

        unet_batch_size: int = 1,
        video_length: Optional[int] = None,
        context_frames: int = -1,
        context_stride: int = 3,
        context_overlap: int = 4,
        context_schedule: str = "uniform",
        clip_skip: int = 1,
        controlnet_type_map: Dict[str, Dict[str,float]] = None,
        controlnet_image_map: Dict[int, Dict[str,Any]] = None,
        controlnet_ref_map: Dict[str, Any] = None,
        controlnet_max_samples_on_vram: int = 999,
        controlnet_max_models_on_vram: int=99,
        controlnet_is_loop: bool=True,
        img2img_map: Dict[str, Any] = None,
        ip_adapter_config_map: Dict[str,Any] = None,
        region_list: List[Any] = None,
        region_condi_list: List[Any] = None,
        interpolation_factor = 1,
        is_single_prompt_mode = False,
        apply_lcm_lora=False,
        gradual_latent_map=None,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            denoising_end (`float`, *optional*):
                When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be
                completed before it is intentionally prematurely terminated. As a result, the returned sample will
                still retain a substantial amount of noise as determined by the discrete timesteps selected by the
                scheduler. The denoising_end parameter should ideally be utilized when this pipeline forms a part of a
                "Mixture of Denoisers" multi-pipeline setup, as elaborated in [**Refining the Image
                Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output)
            guidance_scale (`float`, *optional*, defaults to 5.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
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
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.7):
                Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
                [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
                Guidance rescale factor should fix overexposure when using zero terminal SNR.
            original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
                `original_size` defaults to `(width, height)` if not specified. Part of SDXL's micro-conditioning as
                explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
                `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
                `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                For most cases, `target_size` should be set to the desired height and width of the generated image. If
                not specified it will default to `(width, height)`. Part of SDXL's micro-conditioning as explained in
                section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).

        Examples:

        Returns:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """

        gradual_latent = False
        if gradual_latent_map:
            gradual_latent = gradual_latent_map["enable"]


        logger.info(f"{apply_lcm_lora=}")
        if apply_lcm_lora:
            self.scheduler = LCMScheduler.from_config(self.scheduler.config)

        controlnet_image_map_org = controlnet_image_map

        controlnet_max_models_on_vram = 0
        controlnet_max_samples_on_vram = 0

        multi_uncond_mode = self.lora_map is not None
        logger.info(f"{multi_uncond_mode=}")

        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            "dummy_str",
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

        # 2. Define call parameters
        if False:
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

        batch_size = 1

        sequential_mode = video_length is not None and video_length > context_frames

        device = self._execution_device
        latents_device = torch.device("cpu") if sequential_mode else device

        if ip_adapter_config_map:
            img_enc_path = "data/models/ip_adapter/models/image_encoder/"
            if ip_adapter_config_map["is_plus"]:
                self.ip_adapter = IPAdapterPlusXL(self, img_enc_path, "data/models/ip_adapter/sdxl_models/ip-adapter-plus_sdxl_vit-h.bin", device, 16)
            elif ip_adapter_config_map["is_plus_face"]:
                self.ip_adapter = IPAdapterPlusXL(self, img_enc_path, "data/models/ip_adapter/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin", device, 16)
            else:
                self.ip_adapter = IPAdapterXL(self, img_enc_path, "data/models/ip_adapter/sdxl_models/ip-adapter_sdxl_vit-h.bin", device, 4)
            self.ip_adapter.set_scale( ip_adapter_config_map["scale"] )
        else:
            self.ip_adapter = None


        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )

        prompt_encoder = PromptEncoderSDXL(
            self,
            device,
            device,#latents_device,
            num_videos_per_prompt,
            do_classifier_free_guidance,
            region_condi_list,
            negative_prompt,
            is_single_prompt_mode,
            clip_skip,
            multi_uncond_mode=multi_uncond_mode
        )

        if self.ip_adapter:
            self.ip_adapter.delete_encoder()


        condi_size = prompt_encoder.get_condi_size()


        # 3.5 Prepare controlnet variables

        if self.controlnet_map:
            for i, type_str in enumerate(self.controlnet_map):
                if i < controlnet_max_models_on_vram:
                    self.controlnet_map[type_str].to(device=device, non_blocking=True)


        # controlnet_image_map
        # { 0 : { "type_str" : IMAGE, "type_str2" : IMAGE }  }
        # { "type_str" : { 0 : IMAGE, 15 : IMAGE }  }
        controlnet_image_map= None

        if controlnet_image_map_org:
            controlnet_image_map= {key: {} for key in controlnet_type_map}
            for key_frame_no in controlnet_image_map_org:
                for t, img in controlnet_image_map_org[key_frame_no].items():
                    if isinstance( self.controlnet_map[t], ControlNetLLLite ):
                        img_size = 1
                        do_normalize=True
                    else:
                        img_size = prompt_encoder.get_condi_size()
                        do_normalize=False
                    c_dtype = torch.float16 #self.controlnet_map[t].dtype
                    tmp = self.prepare_image(
                        image=img,
                        width=width,
                        height=height,
                        batch_size=1 * 1,
                        num_images_per_prompt=1,
                        #device=device,
                        device=latents_device,
                        dtype=c_dtype,
                        do_classifier_free_guidance=False,
                        guess_mode=False,
                        do_normalize=do_normalize,
                    )
                    controlnet_image_map[t][key_frame_no] = torch.cat([tmp] * img_size)

            del controlnet_image_map_org
            torch.cuda.empty_cache()

        # { "0_type_str" : { "scales" = [0.1, 0.3, 0.5, 1.0, 0.5, 0.3, 0.1], "frames"=[125, 126, 127, 0, 1, 2, 3] }}
        controlnet_scale_map = {}
        controlnet_affected_list = np.zeros(video_length,dtype = int)

        is_v2v = True

        if controlnet_image_map:
            for type_str in controlnet_image_map:
                for key_frame_no in controlnet_image_map[type_str]:
                    scale_list = controlnet_type_map[type_str]["control_scale_list"]
                    if len(scale_list) > 0:
                        is_v2v = False
                    scale_list = scale_list[0: context_frames]
                    scale_len = len(scale_list)

                    if controlnet_is_loop:
                        frames = [ i%video_length for i in range(key_frame_no-scale_len, key_frame_no+scale_len+1)]

                        controlnet_scale_map[str(key_frame_no) + "_" + type_str] = {
                            "scales" : scale_list[::-1] + [1.0] + scale_list,
                            "frames" : frames,
                        }
                    else:
                        frames = [ i for i in range(max(0, key_frame_no-scale_len), min(key_frame_no+scale_len+1, video_length))]

                        controlnet_scale_map[str(key_frame_no) + "_" + type_str] = {
                            "scales" : scale_list[:key_frame_no][::-1] + [1.0] + scale_list[:video_length-key_frame_no-1],
                            "frames" : frames,
                        }

                    controlnet_affected_list[frames] = 1

        def controlnet_is_affected( frame_index:int):
            return controlnet_affected_list[frame_index]

        def get_controlnet_scale(
                type: str,
                cur_step: int,
                step_length: int,
                ):
            s = controlnet_type_map[type]["control_guidance_start"]
            e = controlnet_type_map[type]["control_guidance_end"]
            keep = 1.0 - float(cur_step / len(timesteps) < s or (cur_step + 1) / step_length > e)

            scale = controlnet_type_map[type]["controlnet_conditioning_scale"]

            return keep * scale

        def get_controlnet_variable(
                type_str: str,
                cur_step: int,
                step_length: int,
                target_frames: List[int],
                ):
            cont_vars = []

            if not controlnet_image_map:
                return None

            if type_str not in controlnet_image_map:
                return None

            for fr, img in controlnet_image_map[type_str].items():

                if fr in target_frames:
                    cont_vars.append( {
                        "frame_no" : fr,
                        "image" : img,
                        "cond_scale" : get_controlnet_scale(type_str, cur_step, step_length),
                        "guess_mode" : controlnet_type_map[type_str]["guess_mode"]
                    } )

            return cont_vars



        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=latents_device)
        if img2img_map:
            timesteps, num_inference_steps = self.get_img2img_timesteps(num_inference_steps, img2img_map["denoising_strength"], latents_device)
            latent_timestep = timesteps[:1].repeat(batch_size * 1)
        else:
            timesteps = self.scheduler.timesteps
            latent_timestep = None

        is_strength_max = True
        if img2img_map:
            is_strength_max = img2img_map["denoising_strength"] == 1.0


        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents_outputs = self.prepare_latents(
            batch_size = 1,
            num_channels_latents=num_channels_latents,
            video_length=video_length,
            height=height,
            width=width,
            dtype=prompt_encoder.get_prompt_embeds_dtype(),
            device=latents_device,
            generator=generator,
            img2img_map=img2img_map,
            timestep=latent_timestep,
            latents=latents,
            is_strength_max=is_strength_max,
            return_noise=True,
            return_image_latents=True,
        )

        latents, noise, image_latents = latents_outputs

        del img2img_map
        torch.cuda.empty_cache()

        # 5.5 Prepare region mask
        region_mask = RegionMask(
            region_list,
            batch_size,
            num_channels_latents,
            video_length,
            height,
            width,
            self.vae_scale_factor,
            prompt_encoder.get_prompt_embeds_dtype(),
            latents_device,
            multi_uncond_mode
        )

        torch.cuda.empty_cache()


        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.5 - Infinite context loop shenanigans
        context_scheduler = get_context_scheduler(context_schedule)
        total_steps = get_total_steps(
            context_scheduler,
            timesteps,
            num_inference_steps,
            latents.shape[2],
            context_frames,
            context_stride,
            context_overlap,
        )

        # 7. Prepare added time ids & embeddings
#        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=prompt_encoder.get_prompt_embeds_dtype(),
        )

        add_time_ids = torch.cat([add_time_ids for c in range(condi_size)], dim=0)
        add_time_ids = add_time_ids.to(device)

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        if False:
            # 7.1 Apply denoising_end
            if denoising_end is not None and type(denoising_end) == float and denoising_end > 0 and denoising_end < 1:
                discrete_timestep_cutoff = int(
                    round(
                        self.scheduler.config.num_train_timesteps
                        - (denoising_end * self.scheduler.config.num_train_timesteps)
                    )
                )
                num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
                timesteps = timesteps[:num_inference_steps]


        logger.info(f"{do_classifier_free_guidance=}")
        logger.info(f"{condi_size=}")

        if self.lora_map:
            self.lora_map.to(device, self.unet.dtype)
        if self.lcm:
            self.lcm.to(device, self.unet.dtype)

        lat_height, lat_width = latents.shape[-2:]

        def gradual_latent_scale(progress):
            if gradual_latent:
                cur = 0.5
                for s in gradual_latent_map["scale"]:
                    v = gradual_latent_map["scale"][s]
                    if float(s) > progress:
                        return cur
                    cur = v
                return cur
            else:
                return 1.0
        def gradual_latent_size(progress):
            if gradual_latent:
                current_ratio = gradual_latent_scale(progress)
                h = int(lat_height * current_ratio) // 8 * 8
                w = int(lat_width * current_ratio) // 8 * 8
                return (h,w)
            else:
                return (lat_height, lat_width)

        def unsharp_mask(img):
            imgf = img.float()
            k = 0.05 # strength
            kernel = torch.FloatTensor([[0,   -k,    0],
                                        [-k, 1+4*k, -k],
                                        [0,   -k,    0]])

            conv_kernel = torch.eye(4)[..., None, None] * kernel[None, None, ...]
            imgf = torch.nn.functional.conv2d(imgf, conv_kernel.to(img.device), padding=1)
            return imgf.to(img.dtype)

        def resize_tensor(ten, size, do_unsharp_mask=False):
            ten = rearrange(ten, "b c f h w -> (b f) c h w")
            ten = torch.nn.functional.interpolate(
                ten.float(), size=size, mode="bicubic", align_corners=False
            ).to(ten.dtype)
            if do_unsharp_mask:
                ten = unsharp_mask(ten)
            return rearrange(ten, "(b f) c h w -> b c f h w", f=video_length)

        if gradual_latent:
            latents = resize_tensor(latents, gradual_latent_size(0))
            reverse_steps = gradual_latent_map["reverse_steps"]
            noise_add_count = gradual_latent_map["noise_add_count"]
            total_steps = ((total_steps/num_inference_steps) * (reverse_steps* (len(gradual_latent_map["scale"].keys()) - 1) )) + total_steps
            total_steps = int(total_steps)

        prev_gradient_latent_size = gradual_latent_size(0)



        with self.progress_bar(total=total_steps) as progress_bar:

            i = 0
            real_i = 0
#            for i, t in enumerate(timesteps):
            while i < len(timesteps):
                t = timesteps[i]

                cur_gradient_latent_size = gradual_latent_size((real_i+1) / len(timesteps))

                if self.lcm:
                    self.lcm.apply(i, len(timesteps))


                noise_pred = torch.zeros(
                    (latents.shape[0] * condi_size, *latents.shape[1:]),
                    device=latents.device,
                    dtype=latents.dtype,
                )
                counter = torch.zeros(
                    (1, 1, latents.shape[2], 1, 1), device=latents.device, dtype=latents.dtype
                )

                # { "0_type_str" : (down_samples, mid_sample)  }
                controlnet_result={}

                def apply_lllite(context: List[int]):
                    for type_str in controlnet_type_map:
                        if not isinstance( self.controlnet_map[type_str] , ControlNetLLLite):
                            continue

                        cont_vars = get_controlnet_variable(type_str, i, len(timesteps), context)
                        if not cont_vars:
                            self.controlnet_map[type_str].set_multiplier(0.0)
                            continue

                        def get_index(l, x):
                            return l.index(x) if x in l else -1

                        zero_img = torch.zeros_like(cont_vars[0]["image"])

                        scales=[0.0 for fr in context]
                        imgs=[zero_img for fr in context]

                        for cont_var in cont_vars:
                            c_fr = cont_var["frame_no"]
                            scale_index = str(c_fr) + "_" + type_str

                            for s_i, fr in enumerate(controlnet_scale_map[scale_index]["frames"]):
                                index = get_index(context, fr)
                                if index != -1:
                                    scales[index] = controlnet_scale_map[scale_index]["scales"][s_i]
                                    imgs[index] = cont_var["image"]

                        scales = [ s * cont_var["cond_scale"] for s in scales ]


                        imgs = torch.cat(imgs).to(device=device, non_blocking=True)

                        key= ".".join(map(str, context))
                        key= type_str + "." + key

                        self.controlnet_map[type_str].to(device=device)
                        self.controlnet_map[type_str].set_cond_image(imgs,key)
                        self.controlnet_map[type_str].set_multiplier(scales)

                def get_controlnet_result(context: List[int] = None):
                    #logger.info(f"get_controlnet_result called {context=}")

                    if controlnet_image_map is None:
                        return None, None

                    hit = False
                    for n in context:
                        if controlnet_is_affected(n):
                            hit=True
                            break
                    if hit == False:
                        return None, None

                    apply_lllite(context)

                    if len(controlnet_result) == 0:
                        return None, None

                    _down_block_res_samples=[]

                    first_down = list(list(controlnet_result.values())[0].values())[0][0]
                    first_mid = list(list(controlnet_result.values())[0].values())[0][1]
                    for ii in range(len(first_down)):
                        _down_block_res_samples.append(
                            torch.zeros(
                                (first_down[ii].shape[0], first_down[ii].shape[1], len(context) ,*first_down[ii].shape[3:]),
                                device=device,
                                dtype=first_down[ii].dtype,
                                ))
                    _mid_block_res_samples =  torch.zeros(
                                    (first_mid.shape[0], first_mid.shape[1], len(context) ,*first_mid.shape[3:]),
                                    device=device,
                                    dtype=first_mid.dtype,
                                    )

                    for fr in controlnet_result:
                        for type_str in controlnet_result[fr]:
                            result = str(fr) + "_" + type_str

                            val = controlnet_result[fr][type_str]
                            cur_down = [
                                    v.to(device = device, dtype=first_down[0].dtype, non_blocking=True) if v.device != device else v
                                    for v in val[0]
                                    ]
                            cur_mid =val[1].to(device = device, dtype=first_mid.dtype, non_blocking=True) if val[1].device != device else val[1]
                            loc =  list(set(context) & set(controlnet_scale_map[result]["frames"]))
                            scales = []

                            for o in loc:
                                for j, f in enumerate(controlnet_scale_map[result]["frames"]):
                                    if o == f:
                                        scales.append(controlnet_scale_map[result]["scales"][j])
                                        break
                            loc_index=[]

                            for o in loc:
                                for j, f in enumerate( context ):
                                    if o==f:
                                        loc_index.append(j)
                                        break

                            mod = torch.tensor(scales).to(device, dtype=cur_mid.dtype)

                            add = cur_mid * mod[None,None,:,None,None]
                            _mid_block_res_samples[:, :, loc_index, :, :] = _mid_block_res_samples[:, :, loc_index, :, :] + add

                            for ii in range(len(cur_down)):
                                add = cur_down[ii] * mod[None,None,:,None,None]
                                _down_block_res_samples[ii][:, :, loc_index, :, :] = _down_block_res_samples[ii][:, :, loc_index, :, :] + add

                    return _down_block_res_samples, _mid_block_res_samples

                def process_controlnet( target_frames: List[int] = None ):
                    #logger.info(f"process_controlnet called {target_frames=}")
                    nonlocal controlnet_result

                    controlnet_samples_on_vram = 0

                    loc =  list(set(target_frames) & set(controlnet_result.keys()))

                    controlnet_result = {key: controlnet_result[key] for key in loc}

                    target_frames = list(set(target_frames) - set(loc))
                    #logger.info(f"-> {target_frames=}")
                    if len(target_frames) == 0:
                        return

                    def sample_to_device( sample ):
                        nonlocal controlnet_samples_on_vram

                        if controlnet_max_samples_on_vram <= controlnet_samples_on_vram:
                            down_samples = [
                                v.to(device = torch.device("cpu"), non_blocking=True) if v.device != torch.device("cpu") else v
                                for v in sample[0] ]
                            mid_sample = sample[1].to(device = torch.device("cpu"), non_blocking=True) if sample[1].device != torch.device("cpu") else sample[1]
                        else:
                            if sample[0][0].device != device:
                                down_samples = [ v.to(device = device, non_blocking=True) for v in sample[0] ]
                                mid_sample = sample[1].to(device = device, non_blocking=True)
                            else:
                                down_samples = sample[0]
                                mid_sample = sample[1]
                            controlnet_samples_on_vram += 1
                        return down_samples, mid_sample


                    for fr in controlnet_result:
                        for type_str in controlnet_result[fr]:
                            controlnet_result[fr][type_str] = sample_to_device(controlnet_result[fr][type_str])

                    for type_str in controlnet_type_map:

                        if isinstance( self.controlnet_map[type_str] , ControlNetLLLite):
                            continue

                        cont_vars = get_controlnet_variable(type_str, i, len(timesteps), target_frames)
                        if not cont_vars:
                            continue

                        org_device = self.controlnet_map[type_str].device
                        if org_device != device:
                            self.controlnet_map[type_str] = self.controlnet_map[type_str].to(device=device, non_blocking=True)

                        for cont_var in cont_vars:
                            frame_no = cont_var["frame_no"]

                            latent_model_input = (
                                latents[:, :, [frame_no]]
                                .to(device)
                                .repeat( prompt_encoder.get_condi_size(), 1, 1, 1, 1)
                            )
                            control_model_input = self.scheduler.scale_model_input(latent_model_input, t)[:, :, 0]
                            controlnet_prompt_embeds, controlnet_add_text_embeds = prompt_encoder.get_current_prompt_embeds([frame_no], latents.shape[2])

                            controlnet_added_cond_kwargs = {"text_embeds": controlnet_add_text_embeds.to(device=device), "time_ids": add_time_ids}

                            cont_var_img = cont_var["image"].to(device=device)

                            if gradual_latent:
                                cur_lat_height, cur_lat_width = latents.shape[-2:]
                                cont_var_img = torch.nn.functional.interpolate(
                                    cont_var_img.float(), size=(cur_lat_height*8, cur_lat_width*8), mode="bicubic", align_corners=False
                                ).to(cont_var_img.dtype)


                            down_samples, mid_sample = self.controlnet_map[type_str](
                                control_model_input,
                                t,
                                encoder_hidden_states=controlnet_prompt_embeds.to(device=device),
                                controlnet_cond=cont_var_img,
                                conditioning_scale=cont_var["cond_scale"],
                                guess_mode=cont_var["guess_mode"],
                                added_cond_kwargs=controlnet_added_cond_kwargs,
                                return_dict=False,
                            )

                            for ii in range(len(down_samples)):
                                down_samples[ii] = rearrange(down_samples[ii], "(b f) c h w -> b c f h w", f=1)
                            mid_sample = rearrange(mid_sample, "(b f) c h w -> b c f h w", f=1)

                            if frame_no not in controlnet_result:
                                controlnet_result[frame_no] = {}

                            controlnet_result[frame_no][type_str] = sample_to_device((down_samples, mid_sample))

                        if org_device != device:
                            self.controlnet_map[type_str] = self.controlnet_map[type_str].to(device=org_device, non_blocking=True)



                for context in context_scheduler(
                    i, num_inference_steps, latents.shape[2], context_frames, context_stride, context_overlap
                ):

                    if self.lora_map:
                        self.lora_map.unapply()


                    if controlnet_image_map:
                        if is_v2v:
                            controlnet_target = context
                        else:
                            controlnet_target = list(range(context[0]-context_frames, context[0])) + context + list(range(context[-1]+1, context[-1]+1+context_frames))
                            controlnet_target = [f%video_length for f in controlnet_target]
                            controlnet_target = list(set(controlnet_target))

                        process_controlnet(controlnet_target)

                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = (
                        latents[:, :, context]
                        .to(device)
                        .repeat(condi_size, 1, 1, 1, 1)
                    )
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    cur_prompt, add_text_embeds = prompt_encoder.get_current_prompt_embeds(context, latents.shape[2])
                    down_block_res_samples,mid_block_res_sample = get_controlnet_result(context)

                    cur_prompt = cur_prompt.to(device=device)
                    add_text_embeds = add_text_embeds.to(device=device)

                    # predict the noise residual
                    #added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                    ts = torch.tensor([t], dtype=latent_model_input.dtype, device=latent_model_input.device)
                    if condi_size > 1:
                        ts = ts.repeat(condi_size)


                    __pred = []

                    for layer_index in range(0, latent_model_input.shape[0], unet_batch_size):

                        if self.lora_map:
                            self.lora_map.apply(layer_index, latent_model_input.shape[0], context[len(context)//2])

                        layer_width = 1 if is_single_prompt_mode else context_frames

                        __lat = latent_model_input[layer_index:layer_index+unet_batch_size]
                        __cur_prompt = cur_prompt[layer_index * layer_width:(layer_index + unet_batch_size)*layer_width]
                        __added_cond_kwargs = {"text_embeds": add_text_embeds[layer_index * layer_width:(layer_index + unet_batch_size)*layer_width], "time_ids": add_time_ids[layer_index:layer_index+unet_batch_size]}

                        __do = []
                        if down_block_res_samples is not None:
                            for do in down_block_res_samples:
                                __do.append(do[layer_index:layer_index+unet_batch_size])
                        else:
                            __do = None

                        __mid = None
                        if mid_block_res_sample is not None:
                            __mid = mid_block_res_sample[layer_index:layer_index+unet_batch_size]

                        pred_layer = self.unet(
                            __lat,
                            ts[layer_index:layer_index+unet_batch_size],
                            encoder_hidden_states=__cur_prompt,
                            cross_attention_kwargs=cross_attention_kwargs,
                            added_cond_kwargs=__added_cond_kwargs,
                            down_block_additional_residuals=__do,
                            mid_block_additional_residual=__mid,
                            return_dict=False,
                        )[0]

                        wh = None

                        if i < len(timesteps) * region_mask.get_crop_generation_rate(layer_index, latent_model_input.shape[0]):
                            #TODO lllite
                            wh, xy_list = region_mask.get_area(layer_index, latent_model_input.shape[0], context)
                            if wh:
                                a_w, a_h = wh
                                __lat_list = []
                                for c_index, xy in enumerate( xy_list ):
                                    a_x, a_y = xy
                                    __lat_list.append( __lat[:,:,[c_index],a_y:a_y+a_h, a_x:a_x+a_w ] )

                                __lat = torch.cat(__lat_list, dim=2)

                                if __do is not None:
                                    __tmp_do = []
                                    for _d, rate in zip(__do, (1,1,1,2,2,2,4,4,4,8,8,8)):
                                        _inner_do_list = []
                                        for c_index, xy in enumerate( xy_list ):
                                            a_x, a_y = xy
                                            _inner_do_list.append(_d[:,:,[c_index],a_y//rate:(a_y+a_h)//rate, a_x//rate:(a_x+a_w)//rate ] )

                                        __tmp_do.append( torch.cat(_inner_do_list, dim=2) )
                                    __do = __tmp_do

                                if __mid is not None:
                                    rate = 8
                                    _mid_list = []
                                    for c_index, xy in enumerate( xy_list ):
                                        a_x, a_y = xy
                                        _mid_list.append( __mid[:,:,[c_index],a_y//rate:(a_y+a_h)//rate, a_x//rate:(a_x+a_w)//rate ] )
                                    __mid = torch.cat(_mid_list, dim=2)

                            crop_pred_layer = self.unet(
                                __lat,
                                ts[layer_index:layer_index+unet_batch_size],
                                encoder_hidden_states=__cur_prompt,
                                cross_attention_kwargs=cross_attention_kwargs,
                                added_cond_kwargs=__added_cond_kwargs,
                                down_block_additional_residuals=__do,
                                mid_block_additional_residual=__mid,
                                return_dict=False,
                            )[0]

                            if wh:
                                a_w, a_h = wh
                                for c_index, xy in enumerate( xy_list ):
                                    a_x, a_y = xy
                                    pred_layer[:,:,[c_index],a_y:a_y+a_h, a_x:a_x+a_w] = crop_pred_layer[:,:,[c_index],:,:]


                        __pred.append( pred_layer )

                    down_block_res_samples = None
                    mid_block_res_sample = None

                    pred = torch.cat(__pred)

                    pred = pred.to(dtype=latents.dtype, device=latents.device)
                    noise_pred[:, :, context] = noise_pred[:, :, context] + pred
                    counter[:, :, context] = counter[:, :, context] + 1
                    progress_bar.update()


                # perform guidance
                noise_size = condi_size
                if do_classifier_free_guidance:
                    noise_pred = (noise_pred / counter)
                    noise_list = list(noise_pred.chunk( noise_size ))

                    if multi_uncond_mode:
                        uc_noise_list = noise_list[:len(noise_list)//2]
                        noise_list = noise_list[len(noise_list)//2:]
                        for n in range(len(noise_list)):
                            noise_list[n] = uc_noise_list[n] + guidance_scale * (noise_list[n] - uc_noise_list[n])
                    else:
                        noise_pred_uncond = noise_list.pop(0)
                        for n in range(len(noise_list)):
                            noise_list[n] = noise_pred_uncond + guidance_scale * (noise_list[n] - noise_pred_uncond)

                    noise_size = len(noise_list)
                    noise_pred = torch.cat(noise_list)


                if gradual_latent:
                    if prev_gradient_latent_size != cur_gradient_latent_size:
                        noise_pred = resize_tensor(noise_pred, cur_gradient_latent_size, True)
                        latents = resize_tensor(latents, cur_gradient_latent_size, True)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if (i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0)) and (
                    callback is not None and (callback_steps is not None and i in callback_steps)
                ):
                    denoised = latents - noise_pred
                    #denoised = self.interpolate_latents(denoised, interpolation_factor, device)
                    video = torch.from_numpy(self.decode_latents(denoised))
                    callback(i, video)

                latents_list = latents.chunk( noise_size )

                tmp_latent = torch.zeros(
                    latents_list[0].shape, device=latents.device, dtype=latents.dtype
                )

                for r_no in range(len(region_list)):
                    mask = region_mask.get_mask( r_no )
                    if gradual_latent:
                        mask = resize_tensor(mask, cur_gradient_latent_size)
                    src = region_list[r_no]["src"]
                    if src == -1:
                        init_latents_proper = image_latents[:1]

                        if i < len(timesteps) - 1:
                            noise_timestep = timesteps[i + 1]
                            init_latents_proper = self.scheduler.add_noise(
                                init_latents_proper, noise, torch.tensor([noise_timestep])
                            )

                        if gradual_latent:
                            lat = resize_tensor(init_latents_proper, cur_gradient_latent_size)
                        else:
                            lat = init_latents_proper
                    else:
                        lat = latents_list[src]

                    tmp_latent = tmp_latent * (1-mask) + lat * mask

                latents = tmp_latent

                init_latents_proper = None
                lat = None
                latents_list = None
                tmp_latent = None

                i+=1
                real_i = max(i, real_i)
                if gradual_latent:
                    if prev_gradient_latent_size != cur_gradient_latent_size:
                        reverse = min(i, reverse_steps)
                        self.scheduler._step_index -= reverse
                        _noise = resize_tensor(noise, cur_gradient_latent_size)
                        for count in range(i, i+noise_add_count):
                            count = min(count,len(timesteps)-1)
                            latents = self.scheduler.add_noise(
                                latents, _noise, torch.tensor([timesteps[count]])
                            )
                        i -= reverse
                        torch.cuda.empty_cache()

                prev_gradient_latent_size = cur_gradient_latent_size


        controlnet_result = None
        torch.cuda.empty_cache()

        # make sure the VAE is in float32 mode, as it overflows in float16
        if self.vae.dtype == torch.float32 and latents.dtype == torch.float16:
            self.upcast_vae()
            latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

        if self.ip_adapter:
            show_gpu("before unload ip_adapter")
            self.ip_adapter.unload()
            self.ip_adapter = None
            torch.cuda.empty_cache()
            show_gpu("after unload ip_adapter")

        self.maybe_free_model_hooks()
        torch.cuda.empty_cache()

        if False:
            if not output_type == "latent":
                latents = rearrange(latents, "b c f h w -> (b f) c h w")
                image = self.vae.decode((latents / self.vae.config.scaling_factor).to(self.vae.device, self.vae.dtype), return_dict=False)[0]
            else:
                raise ValueError(f"{output_type=} not supported")
                image = latents
                return StableDiffusionXLPipelineOutput(images=image)

            #image = self.image_processor.postprocess(image, output_type=output_type)

            # Offload last model to CPU
            if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
                self.final_offload_hook.offload()
            image = ((image + 1) / 2).clamp(0, 1)
            video = rearrange(image, "(b f) c h w -> b c f h w", f=single_model_length).cpu()
            if not return_dict:
                return (video,)
        else:
            # Return latents if requested (this will never be a dict)
            if not output_type == "latent":
                video = self.decode_latents(latents)
            else:
                video = latents

            # Convert to tensor
            if output_type == "tensor":
                video = torch.from_numpy(video)

            # Offload all models
            self.maybe_free_model_hooks()

            if not return_dict:
                return video


        return AnimatePipelineOutput(videos=video)


    # Overrride to properly handle the loading and unloading of the additional text encoder.
    def load_lora_weights(self, pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]], **kwargs):
        # We could have accessed the unet config from `lora_state_dict()` too. We pass
        # it here explicitly to be able to tell that it's coming from an SDXL
        # pipeline.

        state_dict, network_alphas = self.lora_state_dict(
            pretrained_model_name_or_path_or_dict,
            unet_config=self.unet.config,
            **kwargs,
        )
        self.load_lora_into_unet(state_dict, network_alphas=network_alphas, unet=self.unet)

        text_encoder_state_dict = {k: v for k, v in state_dict.items() if "text_encoder." in k}
        if len(text_encoder_state_dict) > 0:
            self.load_lora_into_text_encoder(
                text_encoder_state_dict,
                network_alphas=network_alphas,
                text_encoder=self.text_encoder,
                prefix="text_encoder",
                lora_scale=self.lora_scale,
            )

        text_encoder_2_state_dict = {k: v for k, v in state_dict.items() if "text_encoder_2." in k}
        if len(text_encoder_2_state_dict) > 0:
            self.load_lora_into_text_encoder(
                text_encoder_2_state_dict,
                network_alphas=network_alphas,
                text_encoder=self.text_encoder_2,
                prefix="text_encoder_2",
                lora_scale=self.lora_scale,
            )

    @classmethod
    def save_lora_weights(
        self,
        save_directory: Union[str, os.PathLike],
        unet_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]] = None,
        text_encoder_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]] = None,
        text_encoder_2_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]] = None,
        is_main_process: bool = True,
        weight_name: str = None,
        save_function: Callable = None,
        safe_serialization: bool = True,
    ):
        state_dict = {}

        def pack_weights(layers, prefix):
            layers_weights = layers.state_dict() if isinstance(layers, torch.nn.Module) else layers
            layers_state_dict = {f"{prefix}.{module_name}": param for module_name, param in layers_weights.items()}
            return layers_state_dict

        state_dict.update(pack_weights(unet_lora_layers, "unet"))

        if text_encoder_lora_layers and text_encoder_2_lora_layers:
            state_dict.update(pack_weights(text_encoder_lora_layers, "text_encoder"))
            state_dict.update(pack_weights(text_encoder_2_lora_layers, "text_encoder_2"))

        self.write_lora_layers(
            state_dict=state_dict,
            save_directory=save_directory,
            is_main_process=is_main_process,
            weight_name=weight_name,
            save_function=save_function,
            safe_serialization=safe_serialization,
        )

    def _remove_text_encoder_monkey_patch(self):
        self._remove_text_encoder_monkey_patch_classmethod(self.text_encoder)
        self._remove_text_encoder_monkey_patch_classmethod(self.text_encoder_2)