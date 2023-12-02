import random

from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.models.embeddings import TimestepEmbedding
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler, StableDiffusionPipeline, \
    DDPMScheduler
from diffusers.utils.import_utils import is_xformers_available
from pathlib import Path
from guidance import parse_version
from contextlib import contextmanager
# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from torch.cuda.amp import custom_bwd, custom_fwd

class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

class ToWeightsDType(nn.Module):
    def __init__(self, module: nn.Module, dtype: torch.dtype):
        super().__init__()
        self.module = module
        self.dtype = dtype

    def forward(self, x):
        return self.module(x).to(self.dtype)

class StableDiffusionVSD(nn.Module):
    def __init__(self, device, fp16, vram_O, sd_version='2.1', hf_key=None, t_range=[0.02, 0.98]):
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        print(f'[INFO] loading stable diffusion...')

        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        self.precision_t = torch.float16 if fp16 else torch.float32

        # Create model
        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=self.precision_t)
        if parse_version(torch.__version__) >= parse_version("2"):
            print(
                "[INFO] PyTorch2.0 uses memory efficient attention by default."
            )
        elif not is_xformers_available():
            print(
                "[WARN] xformers is not available, memory efficient attention is not enabled."
            )
        else:
            self.pipe.enable_xformers_memory_efficient_attention()
        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            # pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
        pipe_lora = pipe
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.unet_lora = pipe_lora.unet

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        for p in self.unet_lora.parameters():
            p.requires_grad_(False)

        # FIXME: hard-coded dims
        self.camera_embedding = ToWeightsDType(
            TimestepEmbedding(16, 1280), self.precision_t
        ).to(self.device)
        self.unet_lora.class_embedding = self.camera_embedding

        # set up LoRA layers
        lora_attn_procs = {}
        for name in self.unet_lora.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else self.unet_lora.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = self.unet_lora.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet_lora.config.block_out_channels))[
                    block_id
                ]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet_lora.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
            ).to(device)

        self.unet_lora.set_attn_processor(lora_attn_procs)

        self.lora_layers = AttnProcsLayers(self.unet_lora.attn_processors)
        self.lora_layers._load_state_dict_pre_hooks.clear()
        self.lora_layers._state_dict_hooks.clear()

        # self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler", torch_dtype=self.precision_t)
        self.scheduler = DDPMScheduler.from_pretrained(
            model_key,
            subfolder="scheduler",
            torch_dtype=self.precision_t,
        )
        # used to get prediction type later in training
        self.scheduler_lora = DDPMScheduler.from_pretrained(
            model_key,
            subfolder="scheduler",
            torch_dtype=self.precision_t,
        )

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        print(f'[INFO] loaded stable diffusion!')

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        unet,
        latents,
        t,
        encoder_hidden_states,
        class_labels = None,
        cross_attention_kwargs = None,
    ):
        input_dtype = latents.dtype
        return unet(
            latents.to(self.precision_t),
            t.to(self.precision_t),
            encoder_hidden_states=encoder_hidden_states.to(self.precision_t),
            class_labels=class_labels,
            cross_attention_kwargs=cross_attention_kwargs,
        ).sample.to(input_dtype)
    @torch.no_grad()
    def get_text_embeds(self, prompt):
        # prompt: [str]

        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]

        return embeddings

    @contextmanager
    def disable_unet_class_embedding(self):
        class_embedding = self.unet.class_embedding
        try:
            self.unet.class_embedding = None
            yield
        finally:
            self.unet.class_embedding = class_embedding
    def train_lora(
        self,
        latents,
        text_embeddings,
        camera_condition,
    ):
        B = latents.shape[0]
        latents = latents.detach()

        t = torch.randint(
            int(self.num_train_timesteps * 0.0),
            int(self.num_train_timesteps * 1.0),
            [B],
            dtype=torch.long,
            device=self.device,
        )

        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler_lora.add_noise(latents, noise, t)
        if self.scheduler_lora.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler_lora.config.prediction_type == "v_prediction":
            target = self.scheduler_lora.get_velocity(latents, noise, t)
        else:
            raise ValueError(
                f"Unknown prediction type {self.scheduler_lora.config.prediction_type}"
            )
        _, text_embeddings = text_embeddings.chunk(2)  # conditional text embeddings
        if random.random() < 0.1:
            camera_condition = torch.zeros_like(camera_condition)
        noise_pred = self.forward_unet(
            self.unet_lora,
            noisy_latents,
            t,
            encoder_hidden_states=text_embeddings,
            class_labels=camera_condition.view(B, -1),
            cross_attention_kwargs={"scale": 1.0},
        )
        return F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
    def train_step(self, text_embeddings, pred_rgb, camera_condition, guidance_scale=7.5, guidance_scale_lora=1.0, as_latent=False, grad_scale=1):
        camera_condition = camera_condition.to(self.device)
        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False) * 2 - 1
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)
        B = latents.shape[0]
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, (latents.shape[0],), dtype=torch.long, device=self.device)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            with self.disable_unet_class_embedding():
                # cross_attention_kwargs = {"scale": 0.0} if self.single_model else None
                noise_pred_pretrain = self.forward_unet(
                    self.unet,
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings,
                    cross_attention_kwargs={"scale": 0.0},
                )

            noise_pred_est = self.forward_unet(
                self.unet_lora,
                latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=text_embeddings,
                class_labels=torch.cat(
                    [camera_condition.view(B, -1), torch.zeros_like(camera_condition.view(B, -1))], dim=0
                ),
                cross_attention_kwargs={"scale": 1.0},
            )

            noise_pred_pretrain_uncond, noise_pred_pretrain_text = noise_pred_pretrain.chunk(2)
            noise_pred_pretrain = noise_pred_pretrain_uncond + guidance_scale * (
                    noise_pred_pretrain_text - noise_pred_pretrain_uncond
            )
            # TODO: more general cases
            assert self.scheduler.config.prediction_type == "epsilon"
            if self.scheduler_lora.config.prediction_type == "v_prediction":
                alphas_cumprod = self.scheduler_lora.alphas_cumprod.to(
                    device=latents_noisy.device, dtype=latents_noisy.dtype
                )
                alpha_t = alphas_cumprod[t] ** 0.5
                sigma_t = (1 - alphas_cumprod[t]) ** 0.5

                noise_pred_est = latents_noisy * sigma_t + noise_pred_est * alpha_t

            noise_pred_est_uncond, noise_pred_est_text = noise_pred_est.chunk(2)
            noise_pred_est = noise_pred_est_uncond + guidance_scale_lora * (
                    noise_pred_est_text - noise_pred_est_uncond
            )

        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        grad = grad_scale * w[:, None, None, None] * (noise_pred_pretrain - noise_pred_est)
        grad = torch.nan_to_num(grad)

        vsd_loss = SpecifyGradient.apply(latents, grad)
        lora_loss = self.train_lora(latents, text_embeddings, camera_condition)

        return vsd_loss+lora_loss

    @torch.no_grad()
    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

            # perform guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

        return latents

    def decode_latents(self, latents):

        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    @torch.cuda.amp.autocast(enabled=False)
    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]
        input_dtype = imgs.dtype
        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs.to(self.precision_t)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents.to(input_dtype)

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        pos_embeds = self.get_text_embeds(prompts) # [1, 77, 768]
        neg_embeds = self.get_text_embeds(negative_prompts)
        text_embeds = torch.cat([neg_embeds, pos_embeds], dim=0) # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs





