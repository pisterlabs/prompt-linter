import torch
import torch.nn.functional as F
from diffusers.utils import(
    PIL_INTERPOLATION,
    deprecate,
    is_accelerate_available,
    is_accelerate_version,
    logging,
    replace_example_docstring,
)
import PIL
import numpy as np
from functools import partial
def apply_Correction(pipe):
    pipe.denoising_functions.insert(0, partial(Correction_default, pipe))

    new_function_index = pipe.denoising_step_functions.index(pipe.compute_previous_noisy_sample)
    pipe.denoising_step_functions.insert(new_function_index, partial(correction, pipe))
    

    #reverse
    def remover_Correction():
        pipe.denoising_step_functions.pop(new_function_index)
        pipe.denoising_functions.pop(0)

    pipe.revert_functions.insert(0,remover_Correction)

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
def Correction_default(self,**kwargs):
    if kwargs.get('guidance_rescale') is None:
        kwargs['guidance_rescale']=0.0
    return kwargs
def correction(self, i, t, **kwargs):
    do_classifier_free_guidance=kwargs.get('do_classifier_free_guidance')
    guidance_rescale=kwargs.get('guidance_rescale')
    noise_pred=kwargs.get('noise_pred')
    noise_pred_text=kwargs.get('noise_pred_text')
    if do_classifier_free_guidance and guidance_rescale > 0.0:
        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
        kwargs['noise_pred'] = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)
    return kwargs