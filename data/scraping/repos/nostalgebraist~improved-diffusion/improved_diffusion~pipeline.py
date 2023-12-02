import argparse
import os
import random
from typing import Union, List, Optional

import numpy as np
import torch as th
import torch.nn as nn
import torch.distributed as dist
import blobfile as bf

from einops import rearrange

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    load_config_to_model,
)
from improved_diffusion.image_datasets import load_tokenizer, tokenize

from improved_diffusion.unet import UNetModel
from improved_diffusion.respace import SpacedDiffusion

from improved_diffusion.gaussian_diffusion import SimpleForwardDiffusion, get_named_beta_schedule

import clip


def _strip_space(s):
    return "\n".join([part.strip(" ") for part in s.split("\n")])


def _to_visible(img):
    img = ((img + 1) * 127.5).clamp(0, 255).to(th.uint8)
    img = img.permute(0, 2, 3, 1)
    img = img.contiguous()
    return img


def make_dynamic_threshold_denoised_fn(p):
    def dynamic_threshold_denoised_fn(pred_xstart):
        batch_elems = th.split(pred_xstart, 1, dim=0)
        batch_elems_threshed = []

        for b in batch_elems:
            s = th.quantile(b.abs(), p, dim=None).clamp(min=1)
            bt = b.clamp(min=-s, max=s) / s
            batch_elems_threshed.append(bt)

        pred_xstart_threshed = th.cat(batch_elems_threshed, dim=0)
        return pred_xstart_threshed
    return dynamic_threshold_denoised_fn


def make_dynamic_threshold_denoised_fn_batched(p):
    def dynamic_threshold_denoised_fn(pred_xstart):
        b, c, *spatial = pred_xstart.shape

        flat = pred_xstart.reshape(b, -1)

        s = th.quantile(flat.abs(), p, dim=1).clamp(min=1)
        s = s.reshape((-1, 1, 1, 1))

        pred_xstart_threshed = pred_xstart.clamp(min=-s, max=s) / s

        return pred_xstart_threshed
    return dynamic_threshold_denoised_fn


class SamplingModel(nn.Module):
    def __init__(
        self,
        model: UNetModel,
        diffusion_factory,
        tokenizer,
        timestep_respacing,
        is_super_res=False,
        class_map=None,
    ):
        super().__init__()
        self.model = model
        self.diffusion_factory = diffusion_factory
        self.tokenizer = tokenizer
        self.is_super_res = is_super_res
        self.class_map = class_map

        self.set_timestep_respacing(timestep_respacing)

    def set_timestep_respacing(self, timestep_respacing):
        self.diffusion = self.diffusion_factory(timestep_respacing)

    @staticmethod
    def from_config(checkpoint_path, config_path, timestep_respacing="", class_map=None, clipmod=None, **overrides):
        overrides_ = dict(freeze_capt_encoder=True, use_inference_caching=True, clipmod=clipmod)
        overrides_.update(overrides)

        model, diffusion_factory, tokenizer, is_super_res = load_config_to_model(
            config_path,
            overrides=overrides_
        )
        model.load_state_dict(
            dist_util.load_state_dict(checkpoint_path, map_location="cpu"),
            strict=False
        )
        model.to(dist_util.dev())
        model.eval()

        return SamplingModel(
            model=model,
            diffusion_factory=diffusion_factory,
            tokenizer=tokenizer,
            is_super_res=is_super_res,
            timestep_respacing=timestep_respacing,
            class_map=class_map,
        )

    def sample(
        self,
        text: Union[str, List[str]],
        batch_size: int,
        n_samples: int,
        clip_denoised=True,
        use_ddim=False,
        low_res=None,
        seed=None,
        to_visible=True,
        from_visible=True,
        clf_free_guidance=True,
        guidance_scale=0.,
        txt_drop_string='<mask><mask><mask><mask>',
        capt_drop_string='unknown',
        class_ix_drop=999,
        return_intermediates=False,
        use_prk=False,
        use_plms=False,
        ddim_eta=0.,
        plms_ddim_first_n=0,
        plms_ddim_last_n=None,
        capt: Optional[Union[str, List[str]]]=None,
        yield_intermediates=False,
        guidance_after_step=100000,
        y=None,
        verbose=True,
        noise=None,
        dynamic_threshold_p=0,
        denoised_fn=None,
        noise_cond_ts=0,
        noise_cond_schedule='cosine',
        noise_cond_steps=1000,
        guidance_scale_txt=None,  # if provided and different from guidance_scale, applied using step drop
    ):
        # dist_util.setup_dist()

        if denoised_fn is not None:
            pass  # defer to use
        elif dynamic_threshold_p > 0:
            clip_denoised = False
            # denoised_fn = make_dynamic_threshold_denoised_fn(dynamic_threshold_p)
            denoised_fn = make_dynamic_threshold_denoised_fn_batched(dynamic_threshold_p)

        if self.is_super_res and low_res is None:
            raise ValueError("must pass low_res for super res")

        if isinstance(text, str):
            batch_text = batch_size * [text]
        else:
            if text is not None and len(text) != batch_size:
                raise ValueError(f"got {len(text)} texts for bs {batch_size}")
            batch_text = text

        if text is None and self.model.txt:
            batch_text = batch_size * ['']

        if isinstance(capt, str):
            batch_capt = batch_size * [capt]
        else:
            if capt is not None and len(capt) != batch_size:
                raise ValueError(f"got {len(capt)} capts for bs {batch_size}")
            batch_capt = capt

        batch_y = None
        if isinstance(y, str):
            print(f'in class_map? {y in self.class_map}')
            batch_y = batch_size * [self.class_map.get(y, 0)]
        elif y is not None:
            if isinstance(y, int):
                y = batch_size * [y]
            if len(y) != batch_size:
                raise ValueError(f"got {len(y)} ys for bs {batch_size}")
            if y is not None:
                print(f'in class_map? {[yy in set(self.class_map.values()) for yy in y]}')
            batch_y = y

        n_batches = n_samples // batch_size

        if seed is not None:
            if verbose:
                print(f"setting seed to {seed}")
            th.manual_seed(seed)

        use_prog = yield_intermediates or return_intermediates

        if use_plms:
            sample_fn_base = self.diffusion.plms_sample_loop_progressive if use_prog else self.diffusion.plms_sample_loop
        elif use_prk:
            sample_fn_base = self.diffusion.prk_sample_loop_progressive if use_prog else self.diffusion.prk_sample_loop
        elif use_ddim:
            sample_fn_base = self.diffusion.ddim_sample_loop_progressive if use_prog else self.diffusion.ddim_sample_loop
        else:
            sample_fn_base = self.diffusion.p_sample_loop_progressive if use_prog else self.diffusion.p_sample_loop

        if return_intermediates:
            def sample_fn_(*args, **kwargs):
                sample_array, xstart_array = [], []
                for out in sample_fn_base(*args, **kwargs):
                    sample_array.append(out['sample'])
                    xstart_array.append(out['pred_xstart'])
                return {'sample': sample_array, 'xstart': xstart_array}

            sample_fn = sample_fn_
        else:
            sample_fn = sample_fn_base

        model_kwargs = {}
        sample_fn_kwargs = {}
        if use_ddim or use_prk or use_plms:
            sample_fn_kwargs['eta'] = ddim_eta
        if use_plms:
            sample_fn_kwargs['ddim_first_n'] = plms_ddim_first_n
            sample_fn_kwargs['ddim_last_n'] = plms_ddim_last_n

        if batch_text is not None:
            txt = tokenize(self.tokenizer, batch_text)
            txt = th.as_tensor(txt).to(dist_util.dev())
            model_kwargs["txt"] = txt

        if batch_capt is not None:
            capt = clip.tokenize(batch_capt, truncate=True).to(dist_util.dev())
            model_kwargs["capt"] = capt

        if batch_y is not None:
            batch_y = th.as_tensor(batch_y).to(dist_util.dev())
            model_kwargs["y"] = batch_y

        if clf_free_guidance and (guidance_scale > 0):
            model_kwargs["guidance_scale"] = guidance_scale
            model_kwargs["guidance_after_step"] = guidance_after_step
            model_kwargs["unconditional_model_kwargs"] = {}

            txt_guidance_pdrop = 0.0
            if guidance_scale_txt is not None and guidance_scale_txt != guidance_scale:
                txt_guidance_pkeep = guidance_scale_txt / guidance_scale
                txt_guidance_pdrop = 1 - txt_guidance_pkeep

                nstep = len(self.diffusion.betas)
                txt_guidance_ndrop = int(round(txt_guidance_pdrop * nstep))
                if verbose:
                    print(f"txt_guidance_ndrop: {txt_guidance_ndrop}")

                ixs = list(range(nstep))
                random.shuffle(ixs)
                txt_guidance_drop_ixs = set(ixs[:txt_guidance_ndrop])

                model_kwargs["txt_guidance_drop_ixs"] = txt_guidance_drop_ixs

            if batch_text is not None:
                txt_uncon = batch_size * tokenize(self.tokenizer, [txt_drop_string])
                txt_uncon = th.as_tensor(txt_uncon).to(dist_util.dev())
                model_kwargs["unconditional_model_kwargs"]["txt"] = txt_uncon

            if batch_y is not None:
                y_uncon = batch_size * [class_ix_drop]
                y_uncon = th.as_tensor(y_uncon).to(dist_util.dev())
                model_kwargs["unconditional_model_kwargs"]["y"] = y_uncon

            if batch_capt is not None:
                capt_uncon = clip.tokenize(batch_size * [capt_drop_string], truncate=True).to(dist_util.dev())
                model_kwargs["unconditional_model_kwargs"]["capt"] = capt_uncon

        if self.model.noise_cond:
            model_kwargs["cond_timesteps"] = th.as_tensor(batch_size * [int(noise_cond_ts)]).to(dist_util.dev())
            if "unconditional_model_kwargs" in model_kwargs:
                model_kwargs["unconditional_model_kwargs"]["cond_timesteps"] = model_kwargs["cond_timesteps"]

        if model_kwargs.get("txt_guidance_drop_ixs", set()) != set():
            model_kwargs["unconditional_drop_model_kwargs"] = {
                k: v for k, v in model_kwargs["unconditional_model_kwargs"].items()
            }
            model_kwargs["unconditional_drop_model_kwargs"]["txt"] = model_kwargs["txt"]

        all_low_res = []

        if self.is_super_res:
            # TODO: shape vs. text shape
            # print(f"batch_size: {batch_size} vs low_res shape {low_res.shape}")

            low_res = th.from_numpy(low_res).float()

            if from_visible:
                low_res = low_res / 127.5 - 1.0
                low_res = low_res.permute(0, 3, 1, 2)

            all_low_res = low_res.to(dist_util.dev())
            # print(
            #     f"batch_size: {batch_size} vs low_res kwarg shape {all_low_res.shape}"
            # )

            if self.model.noise_cond and noise_cond_ts > 0:
                betas = get_named_beta_schedule(noise_cond_schedule, noise_cond_steps)
                noise_cond_diffusion = SimpleForwardDiffusion(betas)
                all_low_res = noise_cond_diffusion.q_sample(all_low_res, model_kwargs["cond_timesteps"])

                # set seed again because we care about controlling the main diffusion's randomness
                if seed is not None:
                    if verbose:
                        print(f"re-setting seed to {seed} after noising low res image")
                    th.manual_seed(seed)


        image_channels = self.model.in_channels
        if self.is_super_res:
            image_channels -= all_low_res.shape[1]

        all_images = []
        all_sample_sequences = []
        all_xstart_sequences = []

        wrapped = self.diffusion._wrap_model(self.model)

        while len(all_images) * batch_size < n_samples:
            offset = len(all_images)
            if self.is_super_res:
                model_kwargs["low_res"] = all_low_res[offset : offset + batch_size]

                if "unconditional_model_kwargs" in model_kwargs:
                    model_kwargs["unconditional_model_kwargs"]["low_res"] = model_kwargs["low_res"]

                if "unconditional_drop_model_kwargs" in  model_kwargs:
                    model_kwargs["unconditional_drop_model_kwargs"]["low_res"] = model_kwargs["low_res"]

            sample = sample_fn(
                wrapped,
                (
                    batch_size,
                    image_channels,
                    self.model.image_size,
                    self.model.image_size,
                ),
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                noise=noise,
                **sample_fn_kwargs
            )

            if yield_intermediates:
                def gen():
                    for out in sample:
                        sample_, pred_xstart = out['sample'], out['pred_xstart']
                        if to_visible:
                            sample_ = _to_visible(sample_)
                            pred_xstart = _to_visible(pred_xstart)
                        yield (sample_, pred_xstart)
                return gen()
            elif return_intermediates:
                sample_sequence = sample['sample']
                xstart_sequence = sample['xstart']
                sample = sample_sequence[-1]

            if to_visible:
                sample = _to_visible(sample)
                if return_intermediates:
                    # todo: vectorize
                    sample_sequence = [_to_visible(x) for x in sample_sequence]
                    xstart_sequence = [_to_visible(x) for x in xstart_sequence]

            all_images.append(sample.cpu().numpy())

            if return_intermediates:
                sample_sequence = th.stack(sample_sequence, dim=1)
                xstart_sequence = th.stack(xstart_sequence, dim=1)
                all_sample_sequences.append(sample_sequence.cpu().numpy())
                all_xstart_sequences.append(xstart_sequence.cpu().numpy())

        del wrapped

        self.model.embed_capt_cached.cache_clear()

        all_images = np.concatenate(all_images, axis=0)

        if return_intermediates:
            all_sample_sequences = np.concatenate(all_sample_sequences, axis=0)
            all_xstart_sequences = np.concatenate(all_xstart_sequences, axis=0)

            return all_images, all_sample_sequences, all_xstart_sequences

        return all_images


class SamplingPipeline(nn.Module):
    def __init__(self, base_model: SamplingModel, super_res_model: SamplingModel):
        super().__init__()
        self.base_model = base_model
        self.super_res_model = super_res_model

    def sample(
        self,
        text: Union[str, List[str]],
        batch_size: int,
        n_samples: int,
        clip_denoised=True,
        use_ddim=False,
        ddim_eta=0.,
        clf_free_guidance=True,
        guidance_scale=0.,
        txt_drop_string='<mask><mask><mask><mask>',
        capt_drop_string='unknown',
        low_res=None,
        seed=None,
        batch_size_sres=None,
        n_samples_sres=None,
        clf_free_guidance_sres=False,
        guidance_scale_sres=0.,
        strip_space=True,
        return_both_resolutions=False,
        use_plms=False,
        use_plms_sres=False,
        capt: Optional[Union[str, List[str]]]=None,
        yield_intermediates=False,
        guidance_after_step_base=100000,
        y=None,
        verbose=True,
        noise=None,
        noise_sres=None,
        plms_ddim_first_n=0,
        plms_ddim_first_n_sres=0,
        dynamic_threshold_p=0,
        dynamic_threshold_p_sres=0,
        noise_cond_ts_sres=0,
    ):
        if strip_space:
            if isinstance(text, list):
                text = [_strip_space(s) for s in text]
            else:
                text = _strip_space(text)

        batch_size_sres = batch_size_sres or batch_size
        n_samples_sres = n_samples_sres or n_samples

        def base_sample():
            return self.base_model.sample(
                text,
                batch_size,
                n_samples,
                clip_denoised=clip_denoised,
                use_ddim=use_ddim,
                ddim_eta=ddim_eta,
                clf_free_guidance=clf_free_guidance,
                guidance_scale=guidance_scale,
                txt_drop_string=txt_drop_string,
                seed=seed,
                to_visible=False,
                yield_intermediates=yield_intermediates,
                guidance_after_step=guidance_after_step_base,
                verbose=verbose,
                capt=capt,
                y=y,
                use_plms=use_plms,
                noise=noise,
                plms_ddim_first_n=plms_ddim_first_n,
                dynamic_threshold_p=dynamic_threshold_p,
            )

        def high_res_sample(low_res):
            return self.super_res_model.sample(
                text,
                batch_size_sres,
                n_samples_sres,
                low_res=low_res,
                clip_denoised=clip_denoised,
                use_ddim=use_ddim,
                ddim_eta=ddim_eta,
                clf_free_guidance=clf_free_guidance_sres,
                guidance_scale=guidance_scale_sres,
                txt_drop_string=txt_drop_string,
                seed=seed,
                from_visible=False,
                yield_intermediates=yield_intermediates,
                verbose=verbose,
                use_plms=use_plms_sres,
                noise=noise_sres,
                plms_ddim_first_n=plms_ddim_first_n_sres,
                dynamic_threshold_p=dynamic_threshold_p_sres,
                noise_cond_ts=noise_cond_ts_sres,
            )
        if yield_intermediates:
            return _yield_intermediates(base_sample, high_res_sample)

        low_res = base_sample()
        high_res = high_res_sample(low_res)

        if return_both_resolutions:
            low_res = _to_visible(th.as_tensor(low_res)).cpu().numpy()
            return low_res, high_res
        return high_res

    def sample_with_pruning(
        self,
        text: Union[str, List[str]],
        batch_size: int,
        n_samples: int,
        prune_fn,
        continue_if_all_pruned=True,
        clip_denoised=True,
        use_ddim=False,
        clf_free_guidance=True,
        guidance_scale=0.,
        txt_drop_string='<mask><mask><mask><mask>',
        capt_drop_string='unknown',
        low_res=None,
        seed=None,
        batch_size_sres=None,
        n_samples_sres=None,
        clf_free_guidance_sres=False,
        guidance_scale_sres=0.,
        strip_space=True,
        return_both_resolutions=False,
        use_plms=False,
        use_plms_sres=False,
        plms_ddim_last_n=None,
        plms_ddim_last_n_sres=None,
        capt=None,
        guidance_after_step_base=100000,
        y=None,
        verbose=True,
        plms_ddim_first_n=0,
        plms_ddim_first_n_sres=0,
        dynamic_threshold_p=0,
        dynamic_threshold_p_sres=0,
        noise_cond_ts_sres=0,
    ):
        if strip_space:
            if isinstance(text, list):
                text = [_strip_space(s) for s in text]
            else:
                text = _strip_space(text)

        batch_size_sres = batch_size_sres or batch_size
        n_samples_sres = n_samples_sres or n_samples

        low_res = self.base_model.sample(
            text,
            batch_size,
            n_samples,
            clip_denoised=clip_denoised,
            use_ddim=use_ddim,
            clf_free_guidance=clf_free_guidance,
            guidance_scale=guidance_scale,
            txt_drop_string=txt_drop_string,
            seed=seed,
            use_plms=use_plms,
            to_visible=True,
            plms_ddim_last_n=plms_ddim_last_n,
            plms_ddim_first_n=plms_ddim_first_n,
            capt=capt,
            guidance_after_step=guidance_after_step_base,
            y=y,
            verbose=verbose,
            dynamic_threshold_p=dynamic_threshold_p,
        )
        low_res_pruned, text_pruned = prune_fn(low_res, text)
        if len(low_res_pruned) == 0:
            if continue_if_all_pruned:
                print(
                    f"all {len(low_res)} low res samples would be pruned, skipping prune"
                )
                low_res_pruned = low_res
                text_pruned = text
            else:
                return low_res_pruned

        # n_samples_sres = minimum we're OK sampling
        n_samples_sres = max(n_samples_sres, len(low_res_pruned))

        # TODO: n_samples > batch_size case
        tile_shape = [1] * low_res_pruned.ndim
        tile_shape[0] = n_samples_sres // len(low_res_pruned) + 1
        low_res_pruned = np.tile(low_res_pruned, tile_shape)[:n_samples_sres]

        # text_pruned = (text_pruned * tile_shape[0])[:n_samples_sres]
        text_pruned = text  # TODO: remove 'text_pruned' concept

        high_res = self.super_res_model.sample(
            text_pruned,
            batch_size_sres,
            n_samples_sres,
            low_res=low_res_pruned,
            clip_denoised=clip_denoised,
            use_ddim=use_ddim,
            clf_free_guidance=clf_free_guidance_sres,
            guidance_scale=guidance_scale_sres,
            txt_drop_string=txt_drop_string,
            seed=seed,
            use_plms=use_plms_sres,
            plms_ddim_last_n=plms_ddim_last_n_sres,
            plms_ddim_first_n=plms_ddim_first_n_sres,
            from_visible=True,
            verbose=verbose,
            dynamic_threshold_p=dynamic_threshold_p_sres,
            noise_cond_ts=noise_cond_ts_sres,
        )
        if return_both_resolutions:
            return high_res, low_res
        return high_res

def _yield_intermediates(base_sample, high_res_sample):
    low_res_ = None
    for i, (sample, pred_xstart) in enumerate(base_sample()):
        low_res_ = sample
        yield (_to_visible(sample).cpu().numpy(), _to_visible(pred_xstart).cpu().numpy())
    low_res = low_res_.cpu().numpy()

    for i, (sample, pred_xstart) in enumerate(high_res_sample(low_res)):
        yield (sample.cpu().numpy(), pred_xstart.cpu().numpy())
