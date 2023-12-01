import copy
import os
import random
import shutil

import torch
from accelerate.logging import get_logger
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    UNet2DConditionModel,
)
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerFast
from tqdm import tqdm

from omni_diffusion.data.streaming_dataset import MMStreamingDataset
from .base_trainer import (
    BaseTrainer, compute_snr, compute_vae_encodings, encode_prompts_sd, compute_diffusion_loss
)

logger = get_logger(__name__, log_level="INFO")


class StableDiffusionStepDistiller(BaseTrainer):
    def setup_dataloader(self):
        ds = MMStreamingDataset(
            dataset_path=self.dataset_name_or_path,
            model_name_or_path=self.model_name_or_path,
            model_revision=self.model_revision,
            num_tokenizers=self.num_text_encoders,
            resolution=self.resolution,
            proportion_empty_prompts=self.proportion_empty_prompts,
            center_crop=self.center_crop,
            random_flip=self.random_flip,
        )

        return DataLoader(
            ds,
            batch_size=self.train_batch_size,
            num_workers=self.num_dataset_workers,
            pin_memory=True,
        )

    def setup_noise_scheduler(self) -> tuple[DDIMScheduler, DDIMScheduler]:
        # Load scheduler and models.
        noise_scheduler: DDIMScheduler = DDIMScheduler.from_pretrained(
            self.model_name_or_path,
            revision=self.model_revision,
            subfolder="scheduler",
        )

        if self.prediction_type is not None:
            # set prediction_type of scheduler if defined
            noise_scheduler.register_to_config(prediction_type=self.prediction_type)

        teacher_noise_scheduler: DDIMScheduler = DDIMScheduler.from_config(noise_scheduler.config)

        teacher_noise_scheduler.set_timesteps(self.num_ddim_steps)
        noise_scheduler.set_timesteps(self.num_ddim_steps // 2)

        # Check for terminal SNR in combination with SNR Gamma
        if (
            self.snr_gamma is not None
            and not self.force_snr_gamma
            and (
                hasattr(noise_scheduler.config, "rescale_betas_zero_snr") and noise_scheduler.config.rescale_betas_zero_snr
            )
        ):
            raise ValueError(
                f"The selected noise scheduler for the model `{self.model_name_or_path}` uses rescaled betas for zero SNR.\n"
                "When this configuration is present, the parameter `snr_gamma` may not be used without parameter `force_snr_gamma`.\n"
                "This is due to a mathematical incompatibility between our current SNR gamma implementation, and a sigma value of zero."
            )

        return noise_scheduler, teacher_noise_scheduler

    def fit(self):
        if self.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        accelerator = self.setup_accelerator()

        train_dataloader = self.setup_dataloader()

        noise_scheduler, teacher_noise_scheduler = self.setup_noise_scheduler()

        alphas_cumprod = noise_scheduler.alphas_cumprod.to(accelerator.device)
        sqrt_alphas_cumprod = alphas_cumprod ** 0.5
        sqrt_one_minus_alphas_cumprod = (1 - alphas_cumprod) ** 0.5

        noise_scheduler.timesteps = noise_scheduler.timesteps.to(accelerator.device)
        teacher_noise_scheduler.timesteps = teacher_noise_scheduler.timesteps.to(accelerator.device)

        text_encoders, vae, unet, ema_unet, params_to_optimize = self.setup_models(
            accelerator=accelerator,
            num_text_encoders=self.num_text_encoders,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            revision=self.model_revision,
            subfolder="tokenizer",
            use_fast=True,
        )

        teacher_unet = copy.deepcopy(unet)
        teacher_unet.to(accelerator.device)
        teacher_unet.requires_grad_(False)
        teacher_unet.eval()

        optimizer, lr_scheduler = self.setup_optimizer(params_to_optimize, accelerator=accelerator)

        unet, optimizer, lr_scheduler = accelerator.prepare(
            unet, optimizer, lr_scheduler
        )

        if isinstance(train_dataloader, DataLoader):
            train_dataloader = accelerator.prepare(train_dataloader)

        if self.train_text_encoder:
            text_encoders = [
                accelerator.prepare(text_encoder)
                for text_encoder in text_encoders
            ]

        if accelerator.is_main_process:
            accelerator.init_trackers(
                self.project_name,
                config=self.configs,
            )

        if self.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        else:
            weight_dtype = torch.float32

        total_batch_size = self.train_batch_size * accelerator.num_processes * self.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Instantaneous batch size per device = {self.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.max_steps}")
        global_step = 0

        # Load in the weights and states from a previous checkpoint.
        if self.resume_from_checkpoint is not None:
            if self.resume_from_checkpoint != "latest":
                dir_name = os.path.basename(self.resume_from_checkpoint)
            else:
                # Get the latest checkpoint in the output dir.
                dirs = os.listdir(self.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                dir_name = dirs[-1] if len(dirs) > 0 else None

            if dir_name is None:
                logger.warning(
                    f"Provided `resume_from_checkpoint` ({self.resume_from_checkpoint}) does not exist. "
                    "Training from scratch."
                )
                self.resume_from_checkpoint = None
            else:
                logger.info(f"Loading model from {dir_name}")
                accelerator.load_state(os.path.join(self.output_dir, dir_name))
                global_step = int(dir_name.split("-")[1])

                resume_step = global_step * self.gradient_accumulation_steps

        progress_bar = tqdm(
            range(global_step, self.max_steps),
            desc="Steps",
            disable=not accelerator.is_local_main_process,
        )

        for epoch in range(10000000):
            unet.train()
            if self.train_text_encoder:
                for text_encoder in text_encoders:
                    text_encoder.train()

            train_loss = 0.0
            train_loss_diffusion = 0.0
            train_loss_distillation = 0.0

            # TODO: keep aspec ratio within a batch
            # TODO: quick skip or random block order
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(unet):
                    loss_diffusion, loss_distillation = self.training_step(
                        batch,
                        unet=unet,
                        teacher_unet=teacher_unet,
                        vae=vae,
                        text_encoders=text_encoders,
                        noise_scheduler=noise_scheduler,
                        teacher_noise_scheduler=teacher_noise_scheduler,
                        tokenizer=tokenizer,
                        sqrt_alphas_cumprod=sqrt_alphas_cumprod,
                        sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
                    )

                    loss = loss_diffusion + loss_distillation

                    # Gather the losses across all processes for logging (if we use distributed training).
                    losses = accelerator.gather(loss.detach().repeat(self.train_batch_size))
                    losses_diffusion = accelerator.gather(loss_diffusion.detach().repeat(self.train_batch_size))
                    losses_distillation = accelerator.gather(loss_distillation.detach().repeat(self.train_batch_size))
                    train_loss += losses.mean().item()
                    train_loss_diffusion += losses_diffusion.mean().item()
                    train_loss_distillation += losses_distillation.mean().item()

                    # Backpropagate
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(params_to_optimize, self.gradient_clipping)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    if self.use_ema:
                        ema_unet.step(unet.parameters())

                    progress_bar.update(1)
                    global_step += 1
                    train_loss = train_loss / self.gradient_accumulation_steps
                    train_loss_diffusion = train_loss_diffusion / self.gradient_accumulation_steps
                    train_loss_distillation = train_loss_distillation / self.gradient_accumulation_steps
                    accelerator.log(
                        {
                            "train_loss": train_loss,
                            "train_loss_diffusion": train_loss_diffusion,
                            "train_loss_distillation": train_loss_distillation,
                        },
                        step=global_step,
                    )
                    train_loss = 0.0
                    train_loss_diffusion = 0.0
                    train_loss_distillation = 0.0

                    if global_step % self.checkpointing_every_n_steps == 0:
                        if accelerator.is_main_process:
                            if self.max_checkpoints is not None:
                                checkpoints = os.listdir(self.output_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                                # before we save the new checkpoint, we need to have at _most_ `max_checkpoints - 1` checkpoints
                                if len(checkpoints) >= self.max_checkpoints:
                                    num_to_remove = len(checkpoints) - self.max_checkpoints + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, "
                                        f"removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(self.output_dir, removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint)

                        accelerator.wait_for_everyone()

                        save_path = os.path.join(self.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if accelerator.is_main_process:
                        if global_step % self.validation_every_n_steps == 0:
                            self.validation_step(
                                global_step,
                                accelerator=accelerator,
                                model_name_or_path=self.model_name_or_path,
                                model_revision=self.model_revision,
                                text_encoders=text_encoders,
                                vae=vae,
                                unet=unet,
                                ema_unet=ema_unet,
                                weight_dtype=weight_dtype,
                            )

                if global_step >= self.max_steps:
                    break

        accelerator.end_training()

    def training_step(
        self,
        batch,
        unet: UNet2DConditionModel,
        teacher_unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        text_encoders: list[PreTrainedModel],
        noise_scheduler: DDIMScheduler,
        teacher_noise_scheduler: DDIMScheduler,
        tokenizer: PreTrainedTokenizerFast,
        sqrt_alphas_cumprod: torch.Tensor,
        sqrt_one_minus_alphas_cumprod: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = batch["input_ids"]
        bsz = input_ids.shape[0]

        do_cfg_aware_distillation = random.random() < self.cfg_aware_distillation_prob

        if do_cfg_aware_distillation:
            cfg_scale = torch.rand(bsz, dtype=torch.float32, device=input_ids.device)
            cfg_scale = cfg_scale * (14 - 2) + 2

            uncond_ids = torch.full_like(
                input_ids,
                fill_value=tokenizer.eos_token_id,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            uncond_ids[:, 0] = tokenizer.bos_token_id
            input_ids = torch.cat([input_ids, uncond_ids], dim=0)

        prompt_embeds = encode_prompts_sd(
            input_ids,
            text_encoders=text_encoders,
            train_text_encoder=self.train_text_encoder,
        )

        model_input = compute_vae_encodings(batch["image"], vae)

        if do_cfg_aware_distillation:
            model_input = model_input.repeat(2, 1, 1, 1)

        noise = torch.randn_like(model_input)
        if self.noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += self.noise_offset * torch.randn(
                (model_input.shape[0], model_input.shape[1], 1, 1),
                device=model_input.device,
            )

        # Sample a random timestep for each image
        timestep_indices = torch.randint(
            0, noise_scheduler.num_inference_steps, (model_input.shape[0],),
            dtype=torch.int64,
            device=model_input.device,
        )
        # teacher_timestep_indices = (timestep_indices + 1) * 2 - 1
        teacher_timestep_indices = timestep_indices * 2

        timesteps = noise_scheduler.timesteps[teacher_timestep_indices]
        teacher_timesteps = teacher_noise_scheduler.timesteps[teacher_timestep_indices]
        teacher_timesteps_prev = teacher_noise_scheduler.timesteps[teacher_timestep_indices + 1]

        # Add noise to the model input according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

        model_pred = unet(noisy_model_input, timesteps, prompt_embeds).sample

        loss_diffusion = compute_diffusion_loss(
            model_input=model_input[:bsz] if do_cfg_aware_distillation else model_input,
            noise=noise[:bsz] if do_cfg_aware_distillation else noise,
            timesteps=timesteps[:bsz] if do_cfg_aware_distillation else timesteps,
            model_pred=model_pred[:bsz] if do_cfg_aware_distillation else model_pred,
            noise_scheduler=noise_scheduler,
            snr_gamma=self.snr_gamma,
        )

        if do_cfg_aware_distillation:
            model_pred_text, model_pred_uncond = model_pred.chunk(2, dim=0)
            model_pred = model_pred_uncond + cfg_scale * (model_pred_text - model_pred_uncond)

            model_pred = rescale_noise_cfg(model_pred, model_pred_text, guidance_rescale=self.guidance_rescale)

        student_sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, timesteps, model_input.shape)
        student_sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, timesteps, model_input.shape)

        with torch.no_grad():
            sample = noisy_model_input

            # 2 DDIM Diffusion steps
            for i, t, t_prev in enumerate([
                (teacher_timesteps, teacher_timesteps_prev),
                (teacher_timesteps_prev, None)
            ]):
                teacher_model_pred = teacher_unet(sample, t, prompt_embeds).sample

                if do_cfg_aware_distillation:
                    teacher_model_pred_text, teacher_model_pred_uncond = teacher_model_pred.chunk(2, dim=0)
                    teacher_model_pred = teacher_model_pred_uncond + cfg_scale * (
                        teacher_model_pred_text - teacher_model_pred_uncond
                    )

                    teacher_model_pred = rescale_noise_cfg(
                        teacher_model_pred, teacher_model_pred_text, guidance_rescale=self.guidance_rescale
                    )

                    teacher_model_pred = teacher_model_pred.repeat(2, 1, 1, 1)

                # 2. compute alphas, betas
                sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, sample.shape)
                sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, sample.shape)

                if t_prev is not None:
                    sqrt_alphas_cumprod_t_prev = extract(sqrt_alphas_cumprod, t_prev, sample.shape)
                    sqrt_one_minus_alphas_cumprod_t_prev = extract(sqrt_one_minus_alphas_cumprod, t_prev, sample.shape)

                pred_original_sample = sqrt_alphas_cumprod_t * sample - sqrt_one_minus_alphas_cumprod_t * teacher_model_pred

                # 4. Clip or threshold "predicted x_0"
                if teacher_noise_scheduler.config.thresholding:
                    pred_original_sample = teacher_noise_scheduler._threshold_sample(pred_original_sample)
                elif teacher_noise_scheduler.config.clip_sample:
                    clip_sample_range = teacher_noise_scheduler.config.clip_sample_range
                    pred_original_sample = pred_original_sample.clamp(-clip_sample_range, clip_sample_range)

                if i == 1:
                    pred_epsilon = (
                        noisy_model_input - student_sqrt_alphas_cumprod_t * pred_original_sample
                    ) / student_sqrt_one_minus_alphas_cumprod_t

                    distill_target = student_sqrt_alphas_cumprod_t * pred_epsilon - \
                                    student_sqrt_one_minus_alphas_cumprod_t * pred_original_sample
                else:
                    pred_epsilon = sqrt_alphas_cumprod_t * teacher_model_pred + sqrt_one_minus_alphas_cumprod_t * sample

                    # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
                    pred_sample_direction = sqrt_one_minus_alphas_cumprod_t_prev * pred_epsilon

                    # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
                    prev_sample = sqrt_alphas_cumprod_t_prev * pred_original_sample + pred_sample_direction

                    sample = prev_sample

        assert noise_scheduler.config.prediction_type == "v_prediction"

        if self.snr_gamma is None:
            loss_distillation = F.mse_loss(model_pred.float(), distill_target.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = compute_snr(timesteps, noise_scheduler.alphas_cumprod)
            if noise_scheduler.config.prediction_type == "v_prediction":
                # Velocity objective requires that we add one to SNR values before we divide by them.
                snr = snr + 1
            mse_loss_weights = (
                torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
            )
            # We first calculate the original loss. Then we mean over the non-batch dimensions and
            # rebalance the sample-wise losses with their respective loss weights.
            # Finally, we take the mean of the rebalanced loss.
            loss_distillation = F.mse_loss(model_pred.float(), distill_target.float(), reduction="none")
            loss_distillation = loss_distillation.mean(dim=list(range(1, len(loss_distillation.shape)))) * mse_loss_weights
            loss_distillation = loss_distillation.mean()

        return loss_diffusion, loss_distillation


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


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
