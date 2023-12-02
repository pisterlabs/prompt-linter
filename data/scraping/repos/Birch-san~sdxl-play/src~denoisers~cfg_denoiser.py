from torch import FloatTensor, BoolTensor
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from .denoiser_proto import Denoiser

@dataclass
class CFGDenoiser:
  denoiser: Denoiser
  cross_attention_conds: FloatTensor
  cfg_scale: float = 5.
  added_cond_kwargs: Dict[str, Any] = field(default_factory={})
  cross_attention_mask: Optional[BoolTensor] = None
  guidance_rescale: float = 0.

  def __call__(
    self,
    noised_latents: FloatTensor,
    sigma: FloatTensor,
  ) -> FloatTensor:
    conds_per_sample = 2
    noised_latents_in: FloatTensor = noised_latents.repeat(conds_per_sample, 1, 1, 1)
    del noised_latents
    sigma_in: FloatTensor = sigma.repeat(conds_per_sample)
    del sigma
    # this variable name is a bit leaky - it assumes the model has eps objective
    noise_pred: FloatTensor = self.denoiser.forward(
      input=noised_latents_in,
      sigma=sigma_in,
      encoder_hidden_states=self.cross_attention_conds,
      cross_attention_mask=self.cross_attention_mask,
      added_cond_kwargs=self.added_cond_kwargs,
    )
    uncond, cond = noise_pred.chunk(conds_per_sample)
    del noised_latents_in, sigma_in
    cfged: FloatTensor = uncond + (cond - uncond) * self.cfg_scale
    if self.guidance_rescale == 0:
      return cfged
    return self.rescale_noise(cfged, cond)

  # from diffusers
  # https://github.com/huggingface/diffusers/blob/78922ed7c7e66c20aa95159c7b7a6057ba7d590d/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L62
  def rescale_noise(
    self,
    noise_cfg: FloatTensor,
    noise_pred_text: FloatTensor,
  ) -> FloatTensor:
    """
    https://arxiv.org/abs/2305.08891
    Common Diffusion Noise Schedules and Sample Steps are Flawed
    3.4. Rescale Classifier-Free Guidance
    """
    std_text: FloatTensor = noise_pred_text.std([1,2,3], keepdim=True)
    std_cfg: FloatTensor = noise_cfg.std([1,2,3], keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled: FloatTensor = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    rescaled: FloatTensor = self.guidance_rescale * noise_pred_rescaled + (1 - self.guidance_rescale) * noise_cfg
    return rescaled