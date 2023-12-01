import torch
import PIL
import numpy
import cv2
import torch.nn.functional as F
import tqdm
import time
import os
from torch.multiprocessing import Pool
from PIL import Image
from packaging import version
from typing import List, Dict, Callable, Union, Optional, Any, Tuple
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.utils import PIL_INTERPOLATION
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.models.controlnet import ControlNetModel
from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
    DiffusionPipeline,
)
from diffusers.configuration_utils import FrozenDict
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.utils import deprecate, logging
from transformers import CLIPTextModel, CLIPTokenizer
from .lpw_stable_diffusion import StableDiffusionLongPromptWeightingPipeline, preprocess_image
from ..data_classes.correspondenceMap import CorrespondenceMap
from .. import log_utils as logu
from ..utils import save_latents
from .overlap import OverlapAlgorithm, Overlap, ResizeOverlap, VAEOverlap
from .overlap.johnny_overlap import overlap as temp_overlap

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def preprocess_mask(mask, batch_size, scale_factor=8, blur_radius=0):
    if not isinstance(mask, torch.FloatTensor):
        mask = mask.convert("L")
        w, h = mask.size
        w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8
        mask = mask.resize((w // scale_factor, h // scale_factor), resample=PIL_INTERPOLATION["nearest"])
        mask = numpy.array(mask).astype(numpy.float32)
        if blur_radius:  # Blur mask edges
            if blur_radius % 2 == 0:
                blur_radius += 1  # must be odd
            mask = cv2.GaussianBlur(mask, (blur_radius, blur_radius), sigmaX=0)
        mask /= 255.0
        mask = numpy.tile(mask, (4, 1, 1))
        mask = numpy.vstack([mask[None]] * batch_size)
        mask = 1 - mask  # repaint white, keep black
        mask = torch.from_numpy(mask)
        return mask

    else:
        valid_mask_channel_sizes = [1, 3]
        # if mask channel is fourth tensor dimension, permute dimensions to pytorch standard (B, C, H, W)
        if mask.shape[3] in valid_mask_channel_sizes:
            mask = mask.permute(0, 3, 1, 2)
        elif mask.shape[1] not in valid_mask_channel_sizes:
            raise ValueError(
                f"Mask channel dimension of size in {valid_mask_channel_sizes} should be second or fourth dimension,"
                f" but received mask of shape {tuple(mask.shape)}"
            )
        # (potentially) reduce mask channel dimension from 3 to 1 for broadcasting to latent shape
        mask = mask.mean(dim=1, keepdim=True)
        h, w = mask.shape[-2:]
        h, w = (x - x % 8 for x in (h, w))  # resize to integer multiple of 8
        mask = torch.nn.functional.interpolate(mask, (h // scale_factor, w // scale_factor))
        return mask


class StableDiffusionImg2VideoPipeline(StableDiffusionLongPromptWeightingPipeline, StableDiffusionControlNetInpaintPipeline, DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel],
        scheduler,
        safety_checker,
        feature_extractor,
        requires_safety_checker: bool = True
    ):
        DiffusionPipeline.__init__(self)

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        if isinstance(controlnet, (list, tuple)):
            controlnet = MultiControlNetModel(controlnet)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )
        self.register_to_config(
            requires_safety_checker=requires_safety_checker,
        )

    def check_inputs(
        self,
        prompt,
        images,
        control_images,
        masks,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        controlnet_conditioning_scale=1.0,
        control_guidance_start=0.0,
        control_guidance_end=1.0,
    ):
        if height is not None and height % 8 != 0 or width is not None and width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        # `prompt` needs more sophisticated handling when there are multiple
        # conditionings.
        if isinstance(self.controlnet, MultiControlNetModel):
            if isinstance(prompt, list):
                logger.warning(
                    f"You have {len(self.controlnet.nets)} ControlNets and you have passed {len(prompt)}"
                    " prompts. The conditionings will be fixed across the prompts."
                )

        # Check images and masks
        if images is not None:
            [self.check_image(image, prompt, prompt_embeds) for image in images]
        if masks is not None:
            [self.check_image(mask, prompt, prompt_embeds) for mask in masks]

        # Images, masks and control images should the same length
        if masks is None:
            pass
        elif images is None:
            raise ValueError(
                "If `masks` are provided, `images` must be provided as well. Got `images` = None and `masks` ="
                f" {masks}."
            )
        elif len(images) != len(masks):
            raise ValueError(
                f"`images` and `masks` must have the same length but got `images` {len(images)} != `masks`"
                f" {len(masks)}."
            )

        if images is not None and control_images is not None and len(control_images) != len(images):
            raise ValueError(
                f"`control_images` and `images` must have the same length but got `control_images`"
                f" {len(control_images)} != `images` {len(images)}."
            )

        # Check control images
        is_compiled = hasattr(F, "scaled_dot_product_attention") and isinstance(
            self.controlnet, torch._dynamo.eval_frame.OptimizedModule
        )
        if control_images is None:
            pass
        elif (
            isinstance(self.controlnet, ControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, ControlNetModel)
        ):
            [self.check_image(image, prompt, prompt_embeds) for image in control_images]
        elif (
            isinstance(self.controlnet, MultiControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, MultiControlNetModel)
        ):
            for control_images_ in control_images:
                if not isinstance(control_images_, list):
                    raise TypeError("For multiple controlnets: `images` must be type `list` of `list`")

                # When `image` is a nested list:
                # (e.g. [[canny_image_1, pose_image_1], [canny_image_2, pose_image_2]])
                elif any(isinstance(i, list) for i in control_images_):
                    raise ValueError("A single batch of multiple conditionings are supported at the moment.")
                elif len(control_images_) != len(self.controlnet.nets):
                    raise ValueError(
                        f"For multiple controlnets: `image` must have the same length as the number of controlnets, but got {len(control_images_)} images and {len(self.controlnet.nets)} ControlNets."
                    )

                for control_image_ in control_images_:
                    self.check_image(control_image_, prompt, prompt_embeds)
        else:
            assert False

        # Check `controlnet_conditioning_scale`
        if (
            isinstance(self.controlnet, ControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, ControlNetModel)
        ):
            if not isinstance(controlnet_conditioning_scale, float):
                raise TypeError("For single controlnet: `controlnet_conditioning_scale` must be type `float`.")
        elif (
            isinstance(self.controlnet, MultiControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, MultiControlNetModel)
        ):
            if isinstance(controlnet_conditioning_scale, list):
                if any(isinstance(i, list) for i in controlnet_conditioning_scale):
                    raise ValueError("A single batch of multiple conditionings are supported at the moment.")
            elif isinstance(controlnet_conditioning_scale, list) and len(controlnet_conditioning_scale) != len(
                self.controlnet.nets
            ):
                raise ValueError(
                    "For multiple controlnets: When `controlnet_conditioning_scale` is specified as `list`, it must have"
                    " the same length as the number of controlnets"
                )
        else:
            assert False

        if len(control_guidance_start) != len(control_guidance_end):
            raise ValueError(
                f"`control_guidance_start` has {len(control_guidance_start)} elements, but `control_guidance_end` has {len(control_guidance_end)} elements. Make sure to provide the same number of elements to each list."
            )

        if isinstance(self.controlnet, MultiControlNetModel):
            if len(control_guidance_start) != len(self.controlnet.nets):
                raise ValueError(
                    f"`control_guidance_start`: {control_guidance_start} has {len(control_guidance_start)} elements but there are {len(self.controlnet.nets)} controlnets available. Make sure to provide {len(self.controlnet.nets)}."
                )

        for start, end in zip(control_guidance_start, control_guidance_end):
            if start >= end:
                raise ValueError(
                    f"control guidance start: {start} cannot be larger or equal to control guidance end: {end}."
                )
            if start < 0.0:
                raise ValueError(f"control guidance start: {start} can't be smaller than 0.")
            if end > 1.0:
                raise ValueError(f"control guidance end: {end} can't be larger than 1.0.")

        # TODO: Check correspondence map
        # if correspondence_map is not None:
        #     if isinstance(correspondence_map, CorrespondenceMap):
        #         key = next(iter(correspondence_map.Map.keys))
        #         item = correspondence_map.Map.get(key)
        #         # check an item for the structure[([xpos, ypos], frame_idx)]
        #         if isinstance(item, list) and len(item[0]) == 2 and len(item[0][0]) == 2 and isinstance(item[0][1], int):
        #             pass
        #         else:
        #             raise ValueError(f'Correspondence map item {item} does not have the structure [([xpos, ypos], frame_idx), ...]')

        #     else:
        #         raise TypeError(f"Correspondence map has type {type(correspondence_map)}")

    # Copied from diffusers.pipelines.controlnet.pipeline_controlnet.StableDiffusionControlNetPipeline.prepare_image
    def prepare_control_image(
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
    ):
        image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    def prepare_latents(
        self,
        images,
        timestep,
        num_images_per_prompt,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        num_frames=None,
        same_init_latents=False,
        same_init_noise=False,
    ):
        num_frames = num_frames or len(images)
        if images is None:
            # TODO: This is a bit of a hack, but it works for now.
            batch_size = batch_size * num_images_per_prompt
            shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            if latents is None:
                if same_init_latents:
                    shape = (batch_size, num_channels_latents, 1, height // self.vae_scale_factor, width // self.vae_scale_factor)
                    latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
                    latents = latents.repeat(1, 1, num_frames, 1, 1)
                else:
                    latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            else:
                latents = latents.to(device)

            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma
            return [latents], [None], [None]
        else:
            if same_init_noise:
                shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
                noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

            latents_seq = []
            init_latents_orig_seq = []
            noise_seq = []

            for image in images:
                image = image.to(device=self.device, dtype=dtype)
                init_latent_dist = self.vae.encode(image).latent_dist
                init_latents = init_latent_dist.sample(generator=generator)
                init_latents = self.vae.config.scaling_factor * init_latents

                # Expand init_latents for batch_size and num_images_per_prompt
                init_latents = torch.cat([init_latents] * num_images_per_prompt, dim=0)
                init_latents_orig = init_latents

                # add noise to latents using the timesteps
                if not same_init_noise:
                    noise = randn_tensor(init_latents.shape, generator=generator, device=self.device, dtype=dtype)
                init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
                latents = init_latents

                latents_seq.append(latents)
                init_latents_orig_seq.append(init_latents_orig)
                noise_seq.append(noise)

            return latents_seq, init_latents_orig_seq, noise_seq

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        images: List[PipelineImageInput] = None,
        masks: List[PipelineImageInput] = None,
        control_images: Union[List[PipelineImageInput], List[List[PipelineImageInput]]] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        strength: float = 0.8,
        blur_radius: int = 4,
        num_images_per_prompt: Optional[int] = 1,
        add_predicted_noise: Optional[bool] = False,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_embeddings_multiples: Optional[int] = 3,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        controlnet_conditioning_scale: Union[float, List[float]] = 0.5,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        correspondence_map: Optional[CorrespondenceMap] = None,
        overlap_algorithm: OverlapAlgorithm = None,
        same_init_latents: bool = False,
        same_init_noise: bool = False,

        output_type: Optional[str] = "pil",
        return_dict: bool = True,

        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        is_cancelled_callback: Optional[Callable[[], bool]] = None,
        callback_steps: int = 1,
        callback_kwargs: Optional[Dict[str, Any]] = None,

        view_normal_map: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            images (`List[torch.FloatTensor]` or `List[PIL.Image.Image]`):
                List of frames. Every element, `Image`, or tensor representing an image batch, that will be used as the
                starting point for the process.
            masks (`List[torch.FloatTensor]` or `List[PIL.Image.Image]`):
                `Image`, or tensor representing an image batch, to mask `image`. White pixels in the mask will be
                replaced by noise and therefore repainted, while black pixels will be preserved. If `mask_image` is a
                PIL image, it will be converted to a single channel (luminance) before use. If it's a tensor, it should
                contain one color channel (L) instead of 3, so the expected shape would be `(B, H, W, 1)`.
            control_images (`List[torch.FloatTensor]`, `List[PIL.Image.Image]`, `List[List[torch.FloatTensor]]`, or `List[List[PIL.Image.Image]]`):
                List of control images. Every element, `Image`, or tensor representing an image batch, that will be used as the
                control image for the process. If `control_images` is a list of lists, then every element of the outer list will
                be used as a set of control images for the corresponding element of `images`. This is for MultiControlNet.
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
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
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1.
                `image` will be used as a starting point, adding more noise to it the larger the `strength`. The
                number of denoising steps depends on the amount of noise initially added. When `strength` is 1, added
                noise will be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `image`.
            blur_radius (`int`, *optional*, defaults to 0):
                The radius of the blur filter applied to the mask image.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            add_predicted_noise (`bool`, *optional*, defaults to True):
                Use predicted noise instead of random noise when constructing noisy versions of the original image in
                the reverse diffusion process
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
            max_embeddings_multiples (`int`, *optional*, defaults to `3`):
                The max multiple length of prompt embeddings compared to the max output length of text encoder.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            is_cancelled_callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. If the function returns
                `True`, the inference will be cancelled.
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
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 0.5):
                The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original `unet`. If multiple ControlNets are specified in `init`, you can set
                the corresponding scale as a list.
            guess_mode (`bool`, *optional*, defaults to `False`):
                The ControlNet encoder tries to recognize the content of the input image even if you remove all
                prompts. A `guidance_scale` value between 3.0 and 5.0 is recommended.
            control_guidance_start (`float` or `List[float]`, *optional*, defaults to 0.0):
                The percentage of total steps at which the ControlNet starts applying.
            control_guidance_end (`float` or `List[float]`, *optional*, defaults to 1.0):
                The percentage of total steps at which the ControlNet stops applying.

        Returns:
            `None` if cancelled by `is_cancelled_callback`,
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

        do_img2img = images is not None
        do_inpainting = masks is not None
        do_controlnet = control_images is not None
        do_overlapping = correspondence_map is not None and overlap_algorithm is not None

        num_images = len(images) if do_img2img else 0
        num_masks = len(masks) if do_inpainting else 0
        num_control_images = len(control_images) if do_controlnet else 0
        num_frames = max(num_images, num_masks, num_control_images)

        logu.debug(f"Do Image2Image: {do_img2img}")
        logu.debug(f"Do Inpainting: {do_inpainting}")
        logu.debug(f"Do ControlNet: {do_controlnet}")
        logu.debug(f"Do Overlapping: {do_overlapping}")
        logu.debug(f"Number of frames: {num_frames}")

        # 1. Align format for control guidance
        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = mult * [control_guidance_start], mult * [
                control_guidance_end
            ]

        # 2. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 3. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt=prompt,
            images=images,
            masks=masks,
            control_images=control_images,
            height=height,
            width=width,
            callback_steps=callback_steps,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
        )

        # 4. Define call parameters
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

        if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = guess_mode or global_pool_conditions

        # 5. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            max_embeddings_multiples,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        dtype = prompt_embeds.dtype

        # 6. Preprocess image and mask
        if do_img2img:
            for i, image in enumerate(images):
                if isinstance(image, PIL.Image.Image):
                    image = image.resize((width, height), resample=Image.Resampling.LANCZOS)
                    image = preprocess_image(image, batch_size)
                    image = image.to(device=self.device, dtype=dtype)
                else:
                    image = None
                images[i] = image

        if do_inpainting:
            for i, mask_image in enumerate(masks):
                if isinstance(mask_image, PIL.Image.Image):
                    mask_image = mask_image.resize((width, height), resample=Image.Resampling.LANCZOS)
                    mask = preprocess_mask(mask_image, batch_size, self.vae_scale_factor, blur_radius)
                    mask = mask.to(device=self.device, dtype=dtype)
                    mask = torch.cat([mask] * num_images_per_prompt)
                else:
                    mask = None
                masks[i] = mask

        # TODO: Process correspondence map
        if do_overlapping:
            pass

        # 7. Prepare control image
        if not do_controlnet:
            pass
        elif isinstance(controlnet, ControlNetModel):
            for i, control_image in enumerate(control_images):
                control_image = control_image.resize((width, height), resample=Image.Resampling.LANCZOS)
                control_image = self.prepare_control_image(
                    image=control_image,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )
                control_images[i] = control_image

        elif isinstance(controlnet, MultiControlNetModel):
            all_control_images = []
            for control_image in control_images:

                control_images_ = []

                for control_image_ in control_image:
                    control_image_ = control_image_.resize((width, height), resample=Image.Resampling.LANCZOS)
                    control_image_ = self.prepare_control_image(
                        image=control_image_,
                        width=width,
                        height=height,
                        batch_size=batch_size * num_images_per_prompt,
                        num_images_per_prompt=num_images_per_prompt,
                        device=device,
                        dtype=controlnet.dtype,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        guess_mode=guess_mode,
                    )

                    control_images_.append(control_image_)

                all_control_images.append(control_images_)

            control_images: List[List] = all_control_images
        else:
            assert False

        # 8. Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device, is_text2img=not do_img2img)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 9. Prepare latent variables
        if do_img2img or do_controlnet:
            num_frames = len(images) if do_img2img else 0
            if do_controlnet:
                num_frames = max(num_frames, len(control_images))

            latents_seq, init_latents_orig_seq, noise_seq = self.prepare_latents(
                images,
                latent_timestep,
                num_images_per_prompt,
                batch_size,
                self.unet.config.in_channels,
                height,
                width,
                dtype,
                device,
                generator,
                latents,
                num_frames=num_frames,
                same_init_latents=same_init_latents,
                same_init_noise=same_init_noise,
            )
        else:
            num_frames = 0
            latents_seq, init_latents_orig_seq, noise_seq = self.prepare_latents(
                None,
                latent_timestep,
                num_images_per_prompt,
                batch_size,
                self.unet.config.in_channels,
                height,
                width,
                dtype,
                device,
                generator,
                latents,
                num_frames=num_frames,
                same_init_latents=same_init_latents,
            )

        # 10. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 10.1 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < i or (i + 1) / len(timesteps) > e)
                for i, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

        # 11. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        # Init noise
        save_latents(
            i=0,
            t=0,
            latents=latents_seq,
            prefix="init",
            **callback_kwargs,
        )

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for step, t in enumerate(timesteps):
                # zero value
                if do_controlnet:
                    if isinstance(controlnet_keep[step], list):
                        cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[step])]
                    else:
                        controlnet_cond_scale = controlnet_conditioning_scale
                        if isinstance(controlnet_cond_scale, list):
                            controlnet_cond_scale = controlnet_cond_scale[0]
                        cond_scale = controlnet_cond_scale * controlnet_keep[step]

                for frame_idx in tqdm.tqdm(range(num_frames), desc="Denoising", leave=False):
                    latents_frame = latents_seq[frame_idx]

                    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                    latent_model_input = torch.cat([latents_frame] * 2) if do_classifier_free_guidance else latents_frame
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # controlnet(s) inference
                    if do_controlnet and frame_idx < num_frames:
                        if guess_mode and do_classifier_free_guidance:
                            # Infer ControlNet only for the conditional batch.
                            control_model_input = latents_frame
                            control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                            controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                        else:
                            control_model_input = latent_model_input
                            controlnet_prompt_embeds = prompt_embeds

                        down_block_res_samples, mid_block_res_sample = self.controlnet(
                            control_model_input,
                            t,
                            encoder_hidden_states=controlnet_prompt_embeds,
                            controlnet_cond=control_images[frame_idx],
                            conditioning_scale=cond_scale,
                            guess_mode=guess_mode,
                            return_dict=False,
                        )

                        if guess_mode and do_classifier_free_guidance:
                            # Infered ControlNet only for the conditional batch.
                            # To apply the output of ControlNet to both the unconditional and conditional batches,
                            # add 0 to the unconditional batch to keep it unchanged.
                            down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                            mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

                    else:
                        down_block_res_samples = mid_block_res_sample = None

                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                    ).sample

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_cond, guidance_rescale=guidance_rescale)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents_frame = self.scheduler.step(noise_pred, t, latents_frame, **extra_step_kwargs, return_dict=False)[0]

                    if hasattr(self.scheduler, "_step_index"):
                        self.scheduler._step_index -= 1

                    # handle inpainting
                    if do_inpainting:
                        mask = masks[frame_idx]
                        init_latents_orig = init_latents_orig_seq[frame_idx]
                        noise = noise_seq[frame_idx]
                        # masking
                        if add_predicted_noise:
                            init_latents_proper = self.scheduler.add_noise(
                                init_latents_orig, noise_pred_uncond, torch.tensor([t])
                            )
                        else:
                            init_latents_proper = self.scheduler.add_noise(init_latents_orig, noise, torch.tensor([t]))
                        latents_frame = (init_latents_proper * mask) + (latents_frame * (1 - mask))

                    latents_seq[frame_idx] = latents_frame

                if hasattr(self.scheduler, '_step_index') and self.scheduler._step_index is not None:
                    self.scheduler._step_index += 1

                if do_overlapping:
                    try:
                        save_dir = callback_kwargs.get('save_dir')
                        if isinstance(save_dir, str):
                            save_dir = os.path.join(save_dir, 'overlapped')
                        else:
                            save_dir = save_dir / 'overlapped'
                        save_latents(
                            step, t, latents_seq,
                            save_dir=save_dir,
                            prefix='latents',
                            postfix='before_overlap',
                            decoder=callback_kwargs.get('decoder', 'vae-approx'),
                            vae=self.vae if callback_kwargs.get('decoder', 'vae-approx') == 'vae' else None
                        )
                    except Exception as e:
                        logu.error(f"{e}, latent not saved")

                    latents_seq = overlap_algorithm.__call__(
                        latents_seq,
                        corr_map=correspondence_map,
                        step=step,
                        timestep=t,
                        view_normal_map=view_normal_map,
                    )
                    # latents_seq = temp_overlap(
                    #     latents_seq,
                    #     corr_map=correspondence_map,
                    #     pipe=self,
                    #     step=step,
                    #     timestep=t,
                    #     init_latents_orig_seq=init_latents_orig_seq,
                    #     noise_seq=noise_seq,
                    # )

                    try:
                        save_dir = callback_kwargs.get('save_dir')
                        if isinstance(save_dir, str):
                            save_dir = os.path.join(save_dir, 'overlapped')
                        else:
                            save_dir = save_dir / 'overlapped'
                        save_latents(step, t, latents_seq,
                                     save_dir=save_dir,
                                     prefix='latents',
                                     postfix='after_overlap',
                                     decoder=callback_kwargs.get('decoder', 'vae-approx'),
                                     vae=self.vae if callback_kwargs.get('decoder', 'vae-approx') == 'vae' else None
                                     )
                    except Exception as e:
                        logu.error(f"{e}, latent not saved")

                # call the callback, if provided
                if step == len(timesteps) - 1 or ((step + 1) > num_warmup_steps and (step + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if step % callback_steps == 0:
                        if callback is not None:
                            callback(step, t, latents_seq, **callback_kwargs)
                        if is_cancelled_callback is not None and is_cancelled_callback():
                            return None

        if output_type == "latent":
            images = latents_seq
            # has_nsfw_concept = None
        elif output_type == "pil":
            # 12. Post-processing
            images = [self.decode_latents(latents) for latents in tqdm.tqdm(latents_seq, desc='Decoding latents')]

            # 13. Run safety checker
            # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

            # 14. Convert to PIL
            images = [self.numpy_to_pil(image) for image in images]
        else:
            # 12. Post-processing
            images = [self.decode_latents(latents) for latents in tqdm.tqdm(latents_seq, desc='Decoding latents')]

            # 13. Run safety checker
            # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return images, None

        return StableDiffusionPipelineOutput(images=images, nsfw_content_detected=None)


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
