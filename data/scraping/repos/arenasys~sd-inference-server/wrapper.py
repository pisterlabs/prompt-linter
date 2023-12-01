import torch
import torchvision.transforms as transforms
import PIL
import random
import io
import os
import safetensors.torch
import time
import shutil
import tomesd
import contextlib
import numpy as np
import math
import json
import itertools
import tqdm

DIRECTML_AVAILABLE = False
try:
    import torch_directml
    DIRECTML_AVAILABLE = True
except:
    pass

import prompts
import samplers_k
import samplers_ddpm
import guidance
import utils
import storage
import upscalers
import inference
import convert
import attention
import controlnet
import preview
import segmentation
import merge
import models
import train

DEFAULTS = {
    "strength": 0.75, "sampler": "Euler a", "clip_skip": 1, "eta": 1,
    "hr_upscaler": "Latent (nearest)", "hr_strength": 0.7, "img2img_upscaler": "Lanczos", "mask_blur": 4,
    "attention": "Default", "vram_mode": "Default"
}

TYPES = {
    int: ["width", "height", "steps", "seed", "batch_size", "clip_skip", "mask_blur", "hr_steps", "padding"],
    float: ["scale", "eta", "hr_factor", "hr_eta"],
}

STATIC = ["storage", "device", "device_names", "callback", "last_models_modified", "last_models_config", "dataset", "public"]

SAMPLER_CLASSES = {
    "Euler": samplers_k.Euler,
    "Euler a": samplers_k.Euler_a,
    "DDIM": samplers_ddpm.DDIM,
    "PLMS": samplers_ddpm.PLMS,
    "DPM++ 2M": samplers_k.DPM_2M,
    "DPM++ 2S a": samplers_k.DPM_2S_a,
    "DPM++ SDE": samplers_k.DPM_SDE,
    "DPM++ 2M SDE": samplers_k.DPM_2M_SDE,

    "Euler Karras": samplers_k.Euler_Karras,
    "Euler a Karras": samplers_k.Euler_a_Karras,
    "DPM++ 2M Karras": samplers_k.DPM_2M_Karras,
    "DPM++ 2S a Karras": samplers_k.DPM_2S_a_Karras,
    "DPM++ SDE Karras": samplers_k.DPM_SDE_Karras,
    "DPM++ 2M SDE Karras": samplers_k.DPM_2M_SDE_Karras,

    "Euler Exponential": samplers_k.Euler_Exponential,
    "Euler a Exponential": samplers_k.Euler_a_Exponential,
    "DPM++ 2M Exponential": samplers_k.DPM_2M_Exponential,
    "DPM++ 2S a Exponential": samplers_k.DPM_2S_a_Exponential,
    "DPM++ SDE Exponential": samplers_k.DPM_SDE_Exponential,
    "DPM++ 2M SDE Exponential": samplers_k.DPM_2M_SDE_Exponential,
}

UPSCALERS_LATENT = {
    "Latent (bicubic)": transforms.InterpolationMode.BICUBIC,
    "Latent (bilinear)": transforms.InterpolationMode.BILINEAR,
    "Latent (nearest)": transforms.InterpolationMode.NEAREST,
}

UPSCALERS_PIXEL = {
    "Lanczos": transforms.InterpolationMode.LANCZOS,
    "Bicubic": transforms.InterpolationMode.BICUBIC,
    "Bilinear": transforms.InterpolationMode.BILINEAR,
    "Nearest": transforms.InterpolationMode.NEAREST,
}

CROSS_ATTENTION = {
    "Default": attention.use_optimized_attention,
    "Split": attention.use_split_attention,
    "Doggettx": attention.use_doggettx_attention,
    "Flash": attention.use_flash_attention,
    "Original": attention.use_diffusers_attention,
    "SDP": attention.use_sdp_attention,
    "XFormers": attention.use_xformers_attention
}

FP32_DEVICES = ["1660", "1650", "1630", "T500", "T550", "T600", "MX550", "MX450", "CMP 30HX"]

def format_float(x):
    return f"{x:.4f}".rstrip('0').rstrip('.')

def model_name(x):
    return x.rsplit('.',1)[0].rsplit(os.path.sep,1)[-1]

class GenerationParameters():
    def __init__(self, storage: storage.ModelStorage, device):
        self.storage = storage
        self.device = device

        self.public = False

        self.device_names = []
        for i in range(torch.cuda.device_count()):
            original = torch.cuda.get_device_name(i)
            name = original
            i = 2
            while name in self.device_names:
                name = original + f" ({i})"
            self.device_names += [name]
        
        if DIRECTML_AVAILABLE:
            self.device_names += ["DirectML"]

        self.device_names += ["CPU"]

        self.last_models_modified = False
        self.last_models_config = None

        self.callback = None

    def switch_public(self):
        self.public = True

    def set_status(self, status):
        if self.callback:
            if not self.callback({"type": "status", "data": {"message": status}}):
                raise RuntimeError("Aborted")

    def set_progress(self, progress):
        if self.callback:
            if not self.callback({"type": "progress", "data": progress}):
                raise RuntimeError("Aborted")

    def on_step(self, progress, latents=None):
        step = self.current_step
        total = self.total_steps
        rate = progress["rate"]
        remaining = (total - step) / rate if rate and total else 0

        if "n" in progress:
            self.current_step += 1
        
        progress = {"current": step, "total": total, "rate": rate, "remaining": remaining, "unit": "it/s"}
        
        interval = int(self.preview_interval or 0)
        if latents != None and self.show_preview and step % interval == 0:
            if self.show_preview == "Full":
                images = preview.full_preview(latents, self.vae)
            elif self.show_preview == "Medium":
                images = preview.model_preview(latents, self.vae)
            else:
                images = preview.cheap_preview(latents, self.vae)
            for i in range(len(images)):
                bytesio = io.BytesIO()
                images[i].save(bytesio, format='PNG')
                images[i] = bytesio.getvalue()
            progress["previews"] = images

        self.set_progress(progress)

    def on_download(self, progress):
        if not progress["rate"]:
            self.set_status("Downloading")

        progress = {"current": progress["n"], "total": progress["total"], "rate": (progress["rate"] or 1) / (1024*1024), "unit": "MB/s"}
        self.set_progress(progress)

    def on_decompose(self, progress):
        if not progress["rate"]:
            self.set_status("Decomposing")
            self.last = None

        if self.last != progress["n"]:
            self.last =  progress["n"]
            progress = {"current": progress["n"], "total": progress["total"], "rate": (progress["rate"] or 1), "unit": "it/s"}
            self.set_progress(progress)

    def on_merge(self, progress):
        if not progress["rate"]:
            self.set_status("Merging")
            self.last = None

        if self.last != progress["n"]:
            self.last =  progress["n"]
            progress = {"current": progress["n"], "total": progress["total"], "rate": (progress["rate"] or 1), "unit": "it/s"}
            self.set_progress(progress)

    def on_training_step(self, progress, epoch, losses):       
        rate = progress["rate"] or 1
        total = progress["total"]
        current = progress["n"]

        elapsed = progress["elapsed"]
        remaining = (total - current)/rate

        progress = {"stage": "Training", "current": current, "total": total, "rate": rate, "elapsed": elapsed, "remaining": remaining, "epoch": epoch, "losses": losses}

        if self.callback:
            if not self.callback({"type": "training_progress", "data": progress}):
                raise RuntimeError("Aborted")

    def on_caching_step(self, progress):
        rate = progress["rate"] or 1
        total = progress["total"]
        current = progress["n"]

        elapsed = progress["elapsed"]
        remaining = (total - current)/rate

        progress = {"stage": "Caching", "current": current, "total": total, "rate": rate, "elapsed": elapsed, "remaining": remaining}

        if self.callback:
            if not self.callback({"type": "training_progress", "data": progress}):
                raise RuntimeError("Aborted")

    def on_artifact(self, name, images):
        if self.callback:
            if type(images[0]) == list:
                for j in range(max([len(i) for i in images])):
                    images_data = []
                    for i in range(len(images)):
                        if j < len(images[i]):
                            bytesio = io.BytesIO()
                            images[i][j].save(bytesio, format='PNG')
                            images_data += [bytesio.getvalue()]
                        else:
                            images_data += [None]
                    self.callback({"type": "artifact", "data": {"name": f"{name} {j+1}", "images": images_data}})
            else:
                images_data = []
                for i in images:
                    if i != None:
                        bytesio = io.BytesIO()
                        i.save(bytesio, format='PNG')
                        images_data += [bytesio.getvalue()]
                    else:
                        images_data += [None]
                self.callback({"type": "artifact", "data": {"name": name, "images": images_data}})
        
    def on_complete(self, images, metadata):
        if self.callback:
            self.set_status("Fetching")
            images_data = []
            for i in images:
                bytesio = io.BytesIO()
                i.save(bytesio, format='PNG')
                images_data += [bytesio.getvalue()]
            self.callback({"type": "result", "data": {"images": images_data, "metadata": metadata}})
            
    def reset(self):
        for attr in list(self.__dict__.keys()):
            if not attr in STATIC:
                delattr(self, attr)

    def __getattr__(self, item):
        return None

    def set(self, **kwargs):
        for key, value in kwargs.items():
            if key == "image" or key == "cn_image":
                for i in range(len(value)):
                    if type(value[i]) == bytes or type(value[i]) == bytearray:
                        value[i] = PIL.Image.open(io.BytesIO(value[i]))
                        if value[i].mode == 'RGBA':
                            value[i] = PIL.Image.alpha_composite(PIL.Image.new('RGBA',value[i].size,(0,0,0)), value[i])
                            value[i] = value[i].convert("RGB")
                        if value[i].mode != 'RGB':
                            value[i] = value[i].convert("RGB")
            if key == "mask":
                for i in range(len(value)):
                    if type(value[i]) == bytes or type(value[i]) == bytearray:
                        value[i] = PIL.Image.open(io.BytesIO(value[i]))
                        if value[i].mode == 'RGBA':
                            value[i] = value[i].split()[-1]
                        else:
                            value[i] = value[i].convert("L")
            if key == "area":
                for i in range(len(value)):
                    for j in range(len(value[i])):
                        if type(value[i][j]) == bytes or type(value) == bytearray:
                            value[i][j] = PIL.Image.open(io.BytesIO(value[i][j]))
                            if value[i][j].mode == 'RGBA':
                                value[i][j] = value[i][j].split()[-1]
                            else:
                                value[i][j] = value[i][j].convert("L")
            if key in STATIC:
                continue
            setattr(self, key, value)

    def load_models(self, unet_nets, clip_nets):
        if self.merge_checkpoint_recipe:
            self.merge_lora_sources = merge.merge_checkpoint(self, self.merge_checkpoint_recipe)
        else:
            self.storage.reset_merge(["UNET"])

            device = self.device

            if self.vram_mode == "Minimal":
                device = torch.device("cpu")
                self.storage.clear_vram()

            if not self.unet or type(self.unet) == str:
                self.unet_name = self.unet or self.model
                self.set_status("Loading UNET")
                self.unet = self.storage.get_unet(self.unet_name, device, unet_nets)
            
            if not self.clip or type(self.clip) == str:
                self.clip_name = self.clip or self.model
                self.set_status("Loading CLIP")
                self.clip = self.storage.get_clip(self.clip_name, device, clip_nets)
                self.clip.set_textual_inversions(self.storage.get_embeddings(self.device))
            
            if not self.vae or type(self.vae) == str:
                self.vae_name = self.vae or self.model
                self.set_status("Loading VAE")
                self.vae = self.storage.get_vae(self.vae_name, device)
                self.configure_vae()
        
        self.storage.clear_file_cache()
        
        self.storage.enforce_controlnet_limit(self.cn or [])
        if self.cn:
            self.set_status("Loading ControlNet")
            self.cn_models = [c for c in self.cn]
            self.cn = [self.storage.get_controlnet(cn, self.device, self.on_download) for cn in self.cn]

        if self.hr_upscaler:
            if not self.hr_upscaler in UPSCALERS_LATENT and not self.hr_upscaler in UPSCALERS_PIXEL:
                self.set_status("Loading Upscaler")
                self.upscale_model = self.storage.get_upscaler(self.hr_upscaler, self.device)

        if self.img2img_upscaler:
            if not self.img2img_upscaler in UPSCALERS_LATENT and not self.img2img_upscaler in UPSCALERS_PIXEL:
                self.set_status("Loading Upscaler")
                self.upscale_model = self.storage.get_upscaler(self.img2img_upscaler, self.device)
    
    def configure_vae(self):
        if not self.vae:
            return
        self.vae.enable_slicing()
        if self.tiling_mode == "Enabled":
            self.vae.enable_tiling()
        else:
            self.vae.disable_tiling()

    def need_models(self, unet, vae, clip):
        if self.vram_mode != "Minimal":
            return
        
        if not unet:
            self.storage.unload(self.unet)
        
        if not vae:
            self.storage.unload(self.vae)
        
        if not clip:
            self.storage.unload(self.clip)

        if unet:
            self.storage.load(self.unet, self.device)
            self.unet.determine_type()
        
        if vae:
            self.storage.load(self.vae, self.device)
            self.configure_vae()
        
        if clip:
            self.storage.load(self.clip, self.device)

    def clear_annotators(self):
        allowed = []
        if self.cn_opts:
            allowed += [o["annotator"] for o in self.cn_opts]
        if self.seg_opts:
            allowed += [o["model"] for o in self.seg_opts]
        if self.cn_annotator:
            allowed += self.cn_annotator
        self.storage.clear_annotators(allowed)

    def check_parameters(self):
        for attr, value in DEFAULTS.items():
            if getattr(self, attr) == None:
                setattr(self, attr, value)

        for t in TYPES:
            for attr in TYPES[t]:
                if getattr(self, attr) != None:
                    setattr(self, attr, t(getattr(self, attr)))

        if not self.sampler in SAMPLER_CLASSES:
            raise ValueError(f"unknown sampler: {self.sampler}")

        if (self.width or self.height) and not (self.width and self.height):
            raise ValueError("width and height must both be set")
        
        if self.vram_mode == "Minimal" and self.show_preview == "Full":
            raise ValueError("Full preview is incompatible with minimal VRAM")
        
        if self.public:
            self.device_name = "Default"
            self.vram_mode = "Default"
            if self.merge_lora_recipe or self.merge_checkpoint_recipe:
                raise Exception("Merging is disabled")
        
        if self.prediction_type:
            self.prediction_type = self.prediction_type.lower()

    def set_device(self):
        device = torch.device("cuda")

        if self.public:
            self.device = device
            return

        if self.device_name in self.device_names:
            idx = self.device_names.index(self.device_name)
            if self.device_name == "CPU":
                device = torch.device("cpu")
                self.storage.dtype = torch.float32
            elif self.device_name == "DirectML":
                device = torch_directml.device()
                self.storage.dtype = torch.float16
            else:
                device = torch.device(idx)
                if any([" " + name in self.device_name for name in FP32_DEVICES]):
                    self.storage.dtype = torch.float32
                else:
                    self.storage.dtype = torch.float16
        self.device = device
    
    def set_attention(self):       
        if self.attention and self.attention in CROSS_ATTENTION:
            CROSS_ATTENTION[self.attention](self.device)

    def listify(self, *args):
        if args == None:
            return (None,)
        args = [[a] if type(a) != list else a for a in args]
        return args

    @torch.inference_mode()
    def upscale_latents(self, latents, mode, width, height, seeds):
        if mode in UPSCALERS_LATENT:
            return upscalers.upscale_single(latents, UPSCALERS_LATENT[mode], width//8, height//8)

        images = utils.decode_images(self.vae, latents)
        if mode in UPSCALERS_PIXEL:
            images = upscalers.upscale(images, UPSCALERS_PIXEL[mode], width, height) 
        else:
            images = upscalers.upscale_super_resolution(images, self.upscale_model, width, height)
        self.set_status("Encoding")
        return utils.encode_images(self.vae, seeds, images)

    def upscale_images(self, images, mode, width, height):
        if mode in UPSCALERS_LATENT:
            raise ValueError(f"cannot use latent upscaler")
        elif mode in UPSCALERS_PIXEL:
            images = upscalers.upscale(images, UPSCALERS_PIXEL[mode], width, height)
        else:
            images = upscalers.upscale_super_resolution(images, self.upscale_model, width, height)
        return images

    def get_seeds(self, batch_size):
        (seeds,) = self.listify(self.seed)
        if self.subseed:
            subseeds = []
            if type(self.subseed[0]) in {tuple, list}:
                subseeds = self.subseed
            else:
                subseeds = [self.subseed]
            subseeds = [tuple(s) for s in subseeds]

            for i in range(len(subseeds)):
                a, b = subseeds[i]
                subseeds[i] = (int(a), float(b))
        else:
            subseeds = [(0,0)]

        for i in range(len(seeds)):
            if seeds[i] == -1:
                seeds[i] = random.randrange(2147483646)
        for i in range(len(subseeds)):
            if subseeds[i][0] == -1:
                subseeds[i] = (random.randrange(2147483646), subseeds[i][1])

        if len(seeds) < batch_size:
            last_seed = seeds[-1]
            seeds += [last_seed + i + 1 for i in range(batch_size-len(seeds))]

        if len(subseeds) < batch_size:
            last_seed, last_strength = subseeds[-1]
            subseeds += [(last_seed + i + 1, last_strength) for i in range(batch_size-len(subseeds))]

        return seeds, subseeds
    
    def get_batch_size(self):
        batch_size = max(self.batch_size or 1, 1)
        for i in [self.prompt, self.negative_prompt, self.seeds, self.subseeds, self.image, self.mask, self.area]:
            if i != None and type(i) == list:
                batch_size = max(batch_size, len(i))
        return batch_size
    
    def get_metadata(self, mode, width, height, batch_size, prompts=None, seeds=None, subseeds=None):
        metadata = []

        for i in range(batch_size):
            m = {
                "mode": mode,
                "width": width,
                "height": height
            }

            if subseeds != None:
                sds = []
                strs = []
                active = False
                for s, r in subseeds:
                    if r != 0.0:
                        active = True
                    sds += [str(s)]
                    strs += [format_float(r)]
                if active:
                    m["subseed"] = ", ".join(sds)
                    m["subseed_strength"] = ", ".join(strs)

            if self.eta != DEFAULTS["eta"]:
                m["eta"] = format_float(self.eta)

            if mode in {"txt2img", "img2img"}:
                if len({self.unet_name, self.clip_name, self.vae_name}) == 1:
                    m["model"] = model_name(self.unet_name)
                else:
                    m["UNET"] = model_name(self.unet_name)
                    m["CLIP"] = model_name(self.clip_name)
                    m["VAE"] = model_name(self.vae_name)
                m["prompt"] = ' AND '.join(prompts[i][0])
                m["negative_prompt"] = ' AND '.join(prompts[i][1])
                m["seed"] = seeds[i]
                m["steps"] = self.steps
                m["scale"] = format_float(self.scale)
                m["sampler"] = self.sampler
                m["clip_skip"] = self.clip_skip
                if self.cfg_rescale:
                    m["cfg_rescale"] = format_float(self.cfg_rescale)
                if self.prediction_type:
                    m["prediction_type"] = self.prediction_type.capitalize()

            if mode == "img2img":
                m["strength"] = format_float(self.strength)
                m["img2img_upscaler"] = model_name(self.img2img_upscaler)
                if self.padding:
                    m["padding"] = self.padding
                m["mask_blur"] = self.mask_blur

            if mode == "txt2img":
                if self.hr_factor and self.hr_factor != 1.0:
                    m["hr_factor"] = format_float(self.hr_factor)
                    m["hr_upscaler"] =  model_name(self.hr_upscaler)
                    m["hr_strength"] = format_float(self.hr_strength)
                    if self.hr_steps and self.hr_steps != self.steps:
                        m["hr_steps"] = self.hr_steps
                    if self.hr_sampler and self.hr_sampler != self.sampler:
                        m["hr_sampler"] = self.hr_sampler
                    if self.hr_eta and self.hr_eta != self.eta:
                        m["hr_eta"] = format_float(self.hr_eta)
            
            if mode == "upscale":
                m["img2img_upscaler"] = model_name(self.img2img_upscaler)
                if self.padding:
                    m["padding"] = self.padding
                m["mask_blur"] = self.mask_blur

            if self.merge_lora_recipe:
                m["merge_lora_recipe"] = self.merge_lora_recipe
                m["merge_lora_strength"] = self.merge_lora_strength
                
            if self.merge_checkpoint_recipe:
                m["merge_checkpoint_recipe"] = self.merge_checkpoint_recipe

            metadata += [m]

        return metadata
    
    def attach_networks(self, all_nets, unet_nets, clip_nets, device):
        self.detach_networks()

        static = self.network_mode == "Static"

        if static:
            self.unet.additional.set_strength([unet_nets])
            self.clip.additional.set_strength([clip_nets])

        lora_names = []
        hn_names = []

        for n in all_nets:
            prefix, name = n.split(":",1)
            if prefix == "lora":
                lora_names += [name]
            if prefix == "hypernet":
                hn_names += [name]

        merged_name, keep_models = None, []
        if self.merge_lora_recipe:
            merged_name, keep_models = merge.merge_lora(self, self.merge_lora_recipe)
            lora_names += [merged_name]
            for model in [self.unet.additional, self.clip.additional]:
                model.set_strength_override(f"lora:{merged_name}", self.merge_lora_strength)
        else:
            self.storage.reset_merge(["LoRA"])

        if self.merge_lora_sources:
            keep_models += self.merge_lora_sources

        if lora_names:
            self.set_status("Loading LoRAs")
            self.storage.enforce_network_limit(lora_names + keep_models, "LoRA")
            self.loras = [self.storage.get_lora(name, device) for name in lora_names]

            for i, lora in enumerate(self.loras):
                is_static = static and lora_names[i] != merged_name
                for model in [self.unet.additional, self.clip.additional]:
                    lora.attach(model, is_static)
                if is_static:
                    lora.to("cpu", torch.float16)
        else:
            self.storage.enforce_network_limit(keep_models, "LoRA")

        if hn_names:
            self.set_status("Loading Hypernetworks")
            self.storage.enforce_network_limit(hn_names, "HN")
            self.hns = [self.storage.get_hypernetwork(name, device) for name in hn_names]

            for hn in self.hns:
                hn.attach(self.unet.additional)
        else:
            self.storage.enforce_network_limit([], "HN")

    def detach_networks(self):
        self.unet.additional.clear()
        self.clip.additional.clear()

    def attach_tome(self, HR=False):
        unet = self.unet
        if type(unet) == controlnet.ControlledUNET:
            unet = unet.unet

        ratio = self.hr_tome_ratio if HR else self.tome_ratio

        if ratio:
            tomesd.apply_patch(unet, ratio=ratio)
        else:
            tomesd.remove_patch(unet)

    def get_autocast_context(self, autocast, device):
        if autocast:
            return torch.autocast('cpu' if device == torch.device('cpu') else 'cuda')
        return contextlib.nullcontext()

    def prepare_images(self, inputs, extents, width, height):
        if not inputs:
            return inputs
        outputs = utils.apply_extents(inputs, extents)
        outputs = upscalers.upscale(outputs, transforms.InterpolationMode.NEAREST, width, height)
        return outputs

    @torch.inference_mode()
    def txt2img(self):
        self.set_status("Configuring")
        self.check_parameters()
        self.clear_annotators()
    
        self.set_status("Parsing")
        conditioning = prompts.BatchedConditioningSchedules(self.prompt, self.steps, self.clip_skip)
        initial_networks = conditioning.get_initial_networks() if self.network_mode == "Static" else ({},{})
        all_networks = conditioning.get_all_networks()

        self.set_status("Loading")
        self.set_device()
        self.set_attention()
        self.load_models(*initial_networks)

        self.attach_tome()

        self.set_status("Configuring")
        batch_size = self.get_batch_size()
        device = self.device

        self.need_models(unet=False, vae=False, clip=False)

        if self.cn:
            self.unet = controlnet.ControlledUNET(self.unet, self.cn)
            dtype = self.unet.dtype

            cn_annotators = [o["annotator"] for o in self.cn_opts]
            cn_scales = [o["scale"] for o in self.cn_opts]
            cn_args = [o["args"] for o in self.cn_opts]
            cn_annotator_models = [self.storage.get_controlnet_annotator(a, device, dtype, self.on_download) for a in cn_annotators]
            cn_images = upscalers.upscale(self.cn_image, transforms.InterpolationMode.LANCZOS, self.width, self.height)

            cn_cond, cn_outputs = controlnet.preprocess_control(cn_images, cn_annotators, cn_annotator_models, cn_args, cn_scales)
            if self.keep_artifacts:
                self.on_artifact("Control", [cn_outputs]*batch_size)
            self.unet.set_controlnet_conditioning(cn_cond, device)

        seeds, subseeds = self.get_seeds(batch_size)
        
        self.current_step = 0
        self.total_steps = self.steps

        if self.hr_factor:
            self.hr_steps = self.hr_steps or self.steps
            self.hr_sampler = self.hr_sampler or self.sampler
            self.hr_eta = self.hr_eta or self.eta
            self.total_steps += self.hr_steps

        metadata = self.get_metadata("txt2img", self.width, self.height, batch_size, self.prompt, seeds, subseeds)

        if self.area:
            area = utils.blur_areas(self.area, self.mask_blur, self.mask_expand)
            if self.keep_artifacts:
                self.on_artifact("Area", area)
            area = utils.preprocess_areas(area, self.width, self.height)
        else:
            area = []
        
        self.set_status("Attaching")

        self.attach_networks(all_networks, *initial_networks, device)

        self.set_status("Encoding")
        self.need_models(unet=False, vae=False, clip=True)
        conditioning.encode(self.clip, area)

        self.set_status("Preparing")
        denoiser = guidance.GuidedDenoiser(self.unet, device, conditioning, self.scale, self.cfg_rescale or 0.0, self.prediction_type)
        noise = utils.NoiseSchedule(seeds, subseeds, self.width // 8, self.height // 8, device, self.unet.dtype)
        sampler = SAMPLER_CLASSES[self.sampler](denoiser, self.eta)

        if self.unet.inpainting: #SDv1-Inpainting, SDv2-Inpainting
            self.need_models(unet=False, vae=True, clip=False)
            images = torch.zeros((batch_size, 3, self.height, self.width))
            inpainting_masked, inpainting_masks = utils.encode_inpainting(images, None, self.vae, seeds)
            denoiser.set_inpainting(inpainting_masked, inpainting_masks)
        
        self.need_models(unet=True, vae=False, clip=False)

        self.set_status("Generating")
        with self.get_autocast_context(self.autocast, device):
            latents = inference.txt2img(denoiser, sampler, noise, self.steps, self.on_step)

        self.need_models(unet=False, vae=True, clip=False)

        if not self.hr_factor:
            self.set_status("Decoding")
            images = utils.decode_images(self.vae, latents)
            self.on_complete(images, metadata)
            self.need_models(unet=False, vae=False, clip=False)
            return images

        self.set_status("Preparing")
        if self.keep_artifacts:
            images = utils.decode_images(self.vae, latents)
            self.on_artifact("Base", images)

        width = int(self.width * self.hr_factor)
        height = int(self.height * self.hr_factor)

        sampler = SAMPLER_CLASSES[self.hr_sampler](denoiser, self.hr_eta)
        noise = utils.NoiseSchedule(seeds, subseeds, width // 8, height // 8, device, self.unet.dtype)

        self.need_models(unet=False, vae=False, clip=True)
        conditioning.switch_to_HR(self.hr_steps)

        if self.network_mode == "Dynamic":
            self.set_status("Attaching")
            hr_all_networks = conditioning.get_all_networks()
            if tuple(sorted(hr_all_networks)) != tuple(sorted(all_networks)):
                self.attach_networks(hr_all_networks, *initial_networks, device)

        if self.area:
            area = utils.blur_areas(self.area, self.mask_blur, self.mask_expand)
            area = utils.preprocess_areas(self.area, width, height)

        self.set_status("Encoding")
        conditioning.encode(self.clip, area)

        self.need_models(unet=False, vae=True, clip=False)

        self.set_status("Upscaling")
        latents = self.upscale_latents(latents, self.hr_upscaler, width, height, seeds)

        if self.keep_artifacts:
            images = utils.decode_images(self.vae, latents)
            self.on_artifact("Upscaled", images)

        if self.unet.inpainting: #SDv1-Inpainting, SDv2-Inpainting
            images = torch.zeros((batch_size, 3, height, width))
            denoiser.set_inpainting(*utils.encode_inpainting(images, None, self.vae, seeds))

        self.need_models(unet=True, vae=False, clip=False)

        self.attach_tome(HR=True)

        if self.cn:
            cn_images = upscalers.upscale(self.cn_image, transforms.InterpolationMode.LANCZOS, width, height)
            cn_cond, cn_outputs = controlnet.preprocess_control(cn_images, cn_annotators, cn_annotator_models, cn_args, cn_scales)
            self.unet.set_controlnet_conditioning(cn_cond, device)
            if self.keep_artifacts:
                self.on_artifact("Control HR", [cn_outputs]*batch_size)
        
        denoiser.reset()

        self.set_status("Generating")

        with self.get_autocast_context(self.autocast, device):
            latents = inference.img2img(latents, denoiser, sampler, noise, self.hr_steps, True, self.hr_strength, self.on_step)

        self.set_status("Decoding")
        self.need_models(unet=False, vae=True, clip=False)
        images = utils.decode_images(self.vae, latents)

        self.on_complete(images, metadata)

        self.need_models(unet=False, vae=False, clip=False)
        return images

    @torch.inference_mode()
    def img2img(self):
        if self.tile_size:
            return self.tiled_img2img()
        
        self.set_status("Configuring")
        self.check_parameters()
        self.clear_annotators()

        self.set_status("Parsing")
        conditioning = prompts.BatchedConditioningSchedules(self.prompt, self.steps, self.clip_skip)
        initial_networks = conditioning.get_initial_networks() if self.network_mode == "Static" else ({},{})
        all_networks = conditioning.get_all_networks()

        self.set_status("Loading")
        self.set_device()
        self.set_attention()
        self.load_models(*initial_networks)

        self.attach_tome()
        
        self.set_status("Preparing")
        batch_size = self.get_batch_size()
        device = self.device
        images = self.image
        masks = (self.mask or []).copy()
        width, height = self.width, self.height

        for i in range(len(masks)):
            if masks[i]:
                masks[i] = upscalers.upscale([masks[i]], transforms.InterpolationMode.LANCZOS, images[i].width, images[i].height)[0]
        extents = utils.get_extents(images, masks, self.padding, width, height)
        for i in range(len(masks)):
            if masks[i] == None:
                masks[i] = PIL.Image.new("L", (images[i].width, images[i].height), color=255)

        for i in range(len(self.cn_image or [])):
            if (self.cn_image[i].width, self.cn_image[i].height) == (images[0].width, images[0].height):
                self.cn_image[i] = self.prepare_images([self.cn_image[i]], [extents[0]], width, height)[0]

        original_images = images
        images = utils.apply_extents(images, extents)
        masks = self.prepare_images(masks, extents, width, height)
        if masks:
            original_masks = [None if mask == None else utils.prepare_mask(mask, 0, 0) for mask in masks]
            masks = [None if mask == None else utils.prepare_mask(mask, self.mask_blur, self.mask_expand) for mask in masks]

        seeds, subseeds = self.get_seeds(batch_size)
        metadata = self.get_metadata("img2img",  width, height, batch_size, self.prompt, seeds, subseeds)
        
        if self.cn:
            self.unet = controlnet.ControlledUNET(self.unet, self.cn)
            dtype = self.unet.dtype

            cn_annotators = [o["annotator"] for o in self.cn_opts]
            cn_scales = [o["scale"] for o in self.cn_opts]
            cn_args = [o["args"] for o in self.cn_opts]
            cn_annotator_models = [self.storage.get_controlnet_annotator(a, device, dtype, self.on_download) for a in cn_annotators]
            cn_images = upscalers.upscale(self.cn_image, transforms.InterpolationMode.LANCZOS, width, height)

            cn_cond, cn_outputs = controlnet.preprocess_control(cn_images, cn_annotators, cn_annotator_models, cn_args, cn_scales, masks=masks)

            if self.keep_artifacts:
                self.on_artifact("Control", [cn_outputs]*batch_size)
            self.unet.set_controlnet_conditioning(cn_cond, device)

        if self.area:
            for i in range(len(self.area)):
                if self.mask and self.mask[i] != None:
                    self.area[i] = self.prepare_images(self.area[i], [extents[i]]*len(self.area[i]), width, height)
            self.area = utils.blur_areas(self.area, self.mask_blur, self.mask_expand)
            if self.keep_artifacts:
                self.on_artifact("Area", self.area)
            self.area = utils.preprocess_areas(self.area, width, height)
        else:
            self.area = []

        self.current_step = 0
        self.total_steps = int(self.steps * self.strength) + 1

        self.set_status("Attaching")
        self.attach_networks(all_networks, *initial_networks, device)

        self.set_status("Encoding")
        self.need_models(unet=False, vae=False, clip=True)
        conditioning.encode(self.clip, self.area)

        denoiser = guidance.GuidedDenoiser(self.unet, device, conditioning, self.scale, self.cfg_rescale or 0.0, self.prediction_type)
        noise = utils.NoiseSchedule(seeds, subseeds, width // 8, height // 8, device, self.unet.dtype)
        sampler = SAMPLER_CLASSES[self.sampler](denoiser, self.eta)

        self.set_status("Upscaling")
        self.need_models(unet=False, vae=True, clip=False)
        upscaled_images = self.upscale_images(images, self.img2img_upscaler, width, height)
        if self.keep_artifacts:
            self.on_artifact("Input", upscaled_images)

        self.set_status("Encoding")
        latents = utils.get_latents(self.vae, seeds, upscaled_images)
        original_latents = latents

        if masks:
            self.set_status("Preparing")
            if self.mask_fill == "Noise":
                original_mask_latents = utils.get_masks(device, original_masks)
                latents = latents * ((original_mask_latents * self.strength) + (1-self.strength))
            
            mask_latents = utils.get_masks(device, masks)
            denoiser.set_mask(mask_latents, original_latents)
            if self.keep_artifacts:
                self.on_artifact("Mask", [masks[i] if self.mask[i] else None for i in range(len(masks))])

        if self.unet.inpainting: #SDv1-Inpainting, SDv2-Inpainting
            self.set_status("Preparing")
            inpainting_masked, inpainting_masks = utils.encode_inpainting(upscaled_images, masks or None, self.vae, seeds)
            denoiser.set_inpainting(inpainting_masked, inpainting_masks)

        self.set_status("Generating")

        self.need_models(unet=True, vae=False, clip=False)

        with self.get_autocast_context(self.autocast, device):
            latents = inference.img2img(latents, denoiser, sampler, noise, self.steps, False, self.strength, self.on_step)

        self.set_status("Decoding")
        
        self.need_models(unet=False, vae=True, clip=False)
        
        images = utils.decode_images(self.vae, latents)

        self.need_models(unet=False, vae=False, clip=False)

        if self.mask:
            outputs, masked = utils.apply_inpainting(images, original_images, masks, extents)
            if self.keep_artifacts:
                self.on_artifact("Output", [masked[i] if self.mask[i] else None for i in range(len(masked))])
            images = [outputs[i] if self.mask[i] else images[i] for i in range(len(images))]

        self.on_complete(images, metadata)
        return images
    
    @torch.inference_mode()
    def tiled_img2img(self):

        tile_size = self.tile_size
        tile_upscale = self.tile_upscale
        tile_strength = self.tile_strength

        self.set_status("Configuring")
        self.check_parameters()
        self.clear_annotators()

        self.set_status("Parsing")
        conditioning = prompts.BatchedConditioningSchedules(self.prompt, self.steps, self.clip_skip)
        initial_networks = conditioning.get_initial_networks() if self.network_mode == "Static" else ({},{})
        all_networks = conditioning.get_all_networks()

        self.set_status("Loading")
        self.set_device()
        self.set_attention()

        if tile_strength:
            self.cn = ["Tile"]
        else:
            self.cn = None
        
        self.load_models(*initial_networks)

        self.attach_tome()
        
        self.set_status("Preparing")
        batch_size = self.get_batch_size()
        device = self.device
        images = self.image
        width, height = self.width, self.height

        seeds, subseeds = self.get_seeds(batch_size)
        metadata = self.get_metadata("img2img",  width, height, batch_size, self.prompt, seeds, subseeds)

        if tile_strength:
            self.unet = controlnet.ControlledUNET(self.unet, self.cn)

        self.set_status("Attaching")
        self.attach_networks(all_networks, *initial_networks, device)

        self.set_status("Encoding")
        self.need_models(unet=False, vae=False, clip=True)
        conditioning.encode(self.clip, [])

        denoiser = guidance.GuidedDenoiser(self.unet, device, conditioning, self.scale, self.cfg_rescale or 0.0, self.prediction_type)
        sampler = SAMPLER_CLASSES[self.sampler](denoiser, self.eta)

        self.set_status("Upscaling")
        self.need_models(unet=False, vae=True, clip=False)
        upscaled_images = self.upscale_images(images, self.img2img_upscaler, width, height)

        tile_images, tile_positions, tile_masks = utils.get_tiles(upscaled_images, tile_size, tile_upscale)

        self.set_status("Encoding")

        tile_latents = []
        for i in range(len(tile_images)):
            tile_latents += [[]]
            for j in range(len(tile_images[i])):
                tile_latents[i] += [utils.get_latents(self.vae, [seeds[i]], [tile_images[i][j]])[0]]

        self.set_status("Generating")

        self.need_models(unet=True, vae=False, clip=False)

        self.current_step = 0
        self.total_steps = (int(self.steps * self.strength) + 1) * sum([len(a) for a in tile_latents])

        with self.get_autocast_context(self.autocast, device):
            for i in range(len(tile_latents)):  
                noise = utils.NoiseSchedule([seeds[i]], [subseeds[i]], tile_size // 8, tile_size // 8, device, self.unet.dtype)
                for j in range(len(tile_latents[i])):
                    if tile_strength:
                        cond, _ = controlnet.annotate(tile_images[i][j], None, None, None)
                        self.unet.set_controlnet_conditioning([(tile_strength,cond)], device)
                    tile_latents[i][j] = inference.img2img(tile_latents[i][j], denoiser, sampler, noise, self.steps, False, self.strength, self.on_step)

        self.set_status("Decoding")
        
        self.need_models(unet=False, vae=True, clip=False)
        
        for i in range(len(tile_latents)):
            for j in range(len(tile_latents[i])):
                tile_images[i][j] = utils.decode_images(self.vae, tile_latents[i][j])[0]

        assembled = utils.assemble_tiles(upscaled_images, tile_images, tile_positions, tile_masks)

        self.need_models(unet=False, vae=False, clip=False)

        self.on_complete(assembled, metadata)
        return images
    
    @torch.inference_mode()
    def upscale(self):
        self.set_status("Loading")
        self.set_device()
        self.storage.clear_file_cache()
        self.clear_annotators()

        if not self.img2img_upscaler in UPSCALERS_PIXEL and not self.img2img_upscaler in UPSCALERS_LATENT:
            self.set_status("Loading Upscaler")
            self.upscale_model = self.storage.get_upscaler(self.img2img_upscaler, self.device)

        self.set_status("Configuring")
        self.check_parameters()

        batch_size = self.get_batch_size()

        images = self.image
        original_images = images

        if self.mask:
            masks = self.mask

        width, height = self.width, self.height

        metadata = self.get_metadata("upscale", width, height, batch_size)

        if self.mask:
            self.set_status("Preparing")
            for i in range(len(masks)):
                if masks[i]:
                    masks[i] = upscalers.upscale([masks[i]], transforms.InterpolationMode.LANCZOS, images[i].width, images[i].height)[0]
            extents = utils.get_extents(images, masks, self.padding, width, height)
            for i in range(len(masks)):
                if masks[i] == None:
                    masks[i] = PIL.Image.new("L", (images[i].width, images[i].height), color=255)
            original_images = images
            images = utils.apply_extents(images, extents)
            masks = utils.apply_extents(masks, extents)
            masks = upscalers.upscale(masks, transforms.InterpolationMode.LANCZOS, width, height)
            masks = [utils.prepare_mask(mask, self.mask_blur, self.mask_expand) for mask in masks]

        self.set_status("Upscaling")
        if not self.upscale_model:
            images = upscalers.upscale(images, UPSCALERS_PIXEL[self.img2img_upscaler], width, height)
        else:
            images = upscalers.upscale_super_resolution(images, self.upscale_model, width, height)

        if self.mask:
            outputs, masked = utils.apply_inpainting(images, original_images, masks, extents)
            images = [outputs[i] if self.mask[i] else images[i] for i in range(len(images))]

        self.on_complete(images, metadata)
        return images
    
    @torch.inference_mode()
    def annotate(self):
        self.set_status("Loading")
        self.set_device()
        self.storage.clear_file_cache()
        self.clear_annotators()

        device = self.device
        dtype = self.storage.dtype

        self.set_status("Annotating")
        annotator = self.storage.get_controlnet_annotator(self.cn_annotator[0], device, dtype, self.on_download)
        _, img = controlnet.annotate(self.cn_image[0], self.cn_annotator[0], annotator, self.cn_args[0])

        self.set_status("Fetching")
        bytesio = io.BytesIO()
        img.save(bytesio, format='PNG')
        data = bytesio.getvalue()
        if self.callback:
            if not self.callback({"type": "annotate", "data": {"images": [data]}}):
                raise RuntimeError("Aborted")
    
    def options(self):
        self.storage.find_all()
        data = {"sampler": list(SAMPLER_CLASSES.keys())}
        for k in self.storage.files:
            data[k] = list(self.storage.files[k].keys())

        data["hr_upscaler"] = list(UPSCALERS_LATENT.keys()) + list(UPSCALERS_PIXEL.keys()) + data["SR"]
        data["img2img_upscaler"] = list(UPSCALERS_PIXEL.keys()) + data["SR"]

        available = attention.get_available() 
        data["attention"] = [k for k,v in CROSS_ATTENTION.items() if v in available]

        data["TI"] = list(self.storage.embeddings_files.keys())
        data["device"] = self.device_names

        if self.public:
            data["device"] = ["Default"]

        if self.callback:
            if not self.callback({"type": "options", "data": data}):
                raise RuntimeError("Aborted")
    
    @torch.inference_mode()
    def manage(self):
        self.set_status("Configuring")
        self.check_parameters()

        if self.operation == "build":
            self.build(self.file)
        elif self.operation == "build_lora":
            self.build_lora(self.file)
        elif self.operation == "modify":
            self.modify(self.old_file, self.new_file)
        elif self.operation == "prune":
            self.prune(self.file)
        elif self.operation == "rename":
            self.set_status("Renaming")
            os.makedirs(self.new_file.rsplit(os.path.sep,1)[0], exist_ok=True)
            shutil.move(self.old_file, self.new_file)
        elif self.operation == "move":
            self.set_status("Moving")
            for folder in storage.MODEL_FOLDERS[self.new_folder]:
                destination = os.path.join(self.storage.path, folder)
                if self.new_subfolder:
                    destination = os.path.join(destination, self.new_subfolder)
                if os.path.exists(destination):
                    break
            else:
                raise ValueError(f"failed to find destination")
            source = os.path.join(self.storage.path, self.old_file)
            destination = os.path.join(destination, self.old_file.rsplit(os.path.sep, 1)[-1])
            shutil.move(source, destination)
        else:
            raise ValueError(f"unknown operation: {self.operation}")
        
        if not self.callback({"type": "done", "data": {}}):
            raise RuntimeError("Aborted")
    
    def trash_model(self, model, copy=False, delete=False):
        if delete:
            if os.path.isfile(model):
                os.remove(model)
            else:
                shutil.rmtree(model)
            return

        trash_path = os.path.join(self.storage.path, "TRASH")
        trash = os.path.join(trash_path, model.rsplit(os.path.sep)[-1])
        os.makedirs(trash_path, exist_ok=True)

        if copy:
            if os.path.isfile(model):
                shutil.copy(model, trash)
            else:
                shutil.copytree(model, trash)
        else:
            shutil.move(model, trash_path)

    def prune(self, file):
        is_checkpoint = any(file.startswith(f) for f in storage.MODEL_FOLDERS["SD"])
        is_component = any([e in file for e in {".vae.",".clip.",".unet."}])
        if is_checkpoint and not is_component:
            self.unet, self.vae, self.clip = file, file, file
            old_file = os.path.join(self.storage.path, file)
            new_file = old_file.rsplit(".", 1)[0] + ".safetensors"

            self.build(new_file)

            if new_file != old_file:
                self.trash_model(old_file, delete=True)
        else:
            old_file = file
            new_file = file
            on, old_ext = old_file.rsplit(".",1)
            if not old_ext == "safetensors":
                new_file = on + ".safetensors"
            
            self.modify(old_file, new_file)

    def modify(self, old, new):
        old_file = os.path.join(self.storage.path, old)
        new_file = os.path.join(self.storage.path, new)

        if not new:
            self.set_status("Deleting")
            self.trash_model(old_file, delete=True)
            return

        old_ext = old.rsplit(".",1)[-1]
        new_ext = new.rsplit(".",1)[-1]

        if not new_ext in {"safetensors"}:
            raise ValueError(f"unsuported checkpoint type: {new_ext}. supported types are: safetensors")

        is_checkpoint = any(old.startswith(f) for f in storage.MODEL_FOLDERS["SD"])
        is_component = any([e in old.rsplit(os.path.sep, 1)[-1] for e in {".vae.",".clip.",".unet."}])
        if is_checkpoint and not is_component:
            self.set_status("Converting")
            
            state_dict, metadata = convert.convert(old_file)
            state_dict = convert.revert(metadata["model_type"], state_dict)
            
            safetensors.torch.save_file(state_dict, new_file, metadata)

            if new_file != old_file:
                self.trash_model(old_file, delete=True)
        else:
            self.set_status("Converting")
            
            if old_ext in {"pt", "pth", "ckpt", "bin"}:
                state_dict = utils.load_pickle(old_file)
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
            elif old_ext in {"safetensors"}:
                state_dict = safetensors.torch.load_file(old_file)
            else:
                raise ValueError(f"unknown model type: {old_ext}")
            
            caution = False
            if is_component:
                caution = convert.clean_component(state_dict)
            else:
                for k in list(state_dict.keys()):
                    if new_ext in {"safetensors"} and type(state_dict[k]) != torch.Tensor:
                        del state_dict[k]
                        caution = True
                        continue

            for k in list(state_dict.keys()):
                if state_dict[k].dtype in {torch.float32, torch.float64, torch.bfloat16}:
                    state_dict[k] = state_dict[k].to(torch.float16)
            
            if len(state_dict) == 0:
                raise ValueError(f"conversion failed, empty model after pruning")
            
            if caution:
                self.trash_model(old_file, delete=False)

            safetensors.torch.save_file(state_dict, new_file)

            if not caution and new_file != old_file:
                self.trash_model(old_file, delete=True)

    def build(self, file):
        file_type = file.rsplit(".",1)[-1]
        if not file_type in {"safetensors"}:
            raise ValueError(f"unsuported checkpoint type: {file_type}. supported types are: safetensors")

        initial_networks = ({}, {})
        if self.prompt:
            self.set_status("Parsing")
            self.network_mode = "Static"
            conditioning = prompts.BatchedConditioningSchedules(self.prompt, 1, 1)
            initial_networks = conditioning.get_initial_networks()
            all_networks = conditioning.get_all_networks()
        
        self.set_status("Loading")
        self.set_device()

        self.attention = "Default"
        self.set_attention()
        
        self.load_models(*initial_networks)

        device = self.unet.device

        if self.prompt:
            self.set_status("Attaching")
            self.attach_networks(all_networks, *initial_networks, device)
        
        self.set_status("Building")

        state_dict = {}

        model_type = self.unet.model_type

        metadata = {
            "model_type": model_type,
            "prediction_type": self.unet.prediction_type,
            "model_variant": self.unet.model_variant 
        }

        if self.clip.model_type != model_type:
            raise ValueError(f"UNET and CLIP are incompatible")

        comp_state_dict = self.unet.state_dict()
        comp_prefix = self.unet.model_type + ".UNET."
        for k in comp_state_dict:
            state_dict[comp_prefix+k] = comp_state_dict[k]

        comp_state_dict = self.clip.state_dict()
        comp_prefix = self.clip.model_type + ".CLIP."
        for k in comp_state_dict:
            state_dict[comp_prefix+k] = comp_state_dict[k]

        comp_state_dict = self.vae.state_dict()
        comp_prefix = self.vae.model_type + ".VAE."
        for k in comp_state_dict:
            state_dict[comp_prefix+k] = comp_state_dict[k]
        del comp_state_dict

        state_dict = convert.revert(model_type, state_dict)

        self.set_status(f"Saving")
        if not self.storage.path in file:
            if os.path.sep in file:
                file = os.path.join(self.storage.path, file)
            else:
                file = os.path.join(self.storage.get_folder("SD"), file)

        safetensors.torch.save_file(state_dict, file, metadata)

    def build_lora(self, file):
        file_type = file.rsplit(".",1)[-1]
        if not file_type in {"safetensors"}:
            raise ValueError(f"unsuported checkpoint type: {file_type}. supported types are: safetensors")

        self.set_status("Loading")
        self.set_device()

        merged_name, _ = merge.merge_lora(self, self.merge_lora_recipe)
        lora = self.storage.get_lora(merged_name, torch.device("cpu"))
        state_dict = lora.state_dict()

        self.set_status(f"Saving")
        if not self.storage.path in file:
            if os.path.sep in file:
                file = os.path.join(self.storage.path, file)
            else:
                file = os.path.join(self.storage.get_folder("LoRA"), file)

        safetensors.torch.save_file(state_dict, file)

    def segmentation(self):
        self.set_status("Configuring")
        self.set_device()
        self.clear_annotators()

        opts = self.seg_opts[0]

        img = self.image[0].convert('RGB')
        inv = img.copy()

        if opts.get("points", []) and opts.get("labels", []):
            points = np.array(opts["points"])
            labels = np.array(opts["labels"])
        else:
            raise RuntimeError("Segment Anything requires points")
        
        self.set_status("Loading")
        model = self.storage.get_segmentation_annotator(opts["model"], self.device, self.on_download)

        self.set_status("Segmenting")
        mask, inv_mask = segmentation.segment(model, img, points, labels)

        img.putalpha(mask)
        inv.putalpha(inv_mask)

        self.on_artifact("Inverse", [inv])
        self.on_artifact("Mask", [mask])

        bytesio = io.BytesIO()
        img.save(bytesio, format='PNG')
        data = bytesio.getvalue()
        if self.callback:
            if not self.callback({"type": "segmentation", "data": {"images": [data]}}):
                raise RuntimeError("Aborted")

    def train_lora(self):
        import diffusers

        self.set_status("Configuring")
        self.check_parameters()
        self.storage.reset()
        self.set_device()

        steps = self.steps
        learning_schedule = self.learning_schedule
        learning_rate = self.learning_rate
        batch_size = self.batch_size
        image_size = self.image_size
        clip_skip = self.clip_skip

        device = self.device
        dtype = torch.float16

        self.set_status("Loading")

        model = os.path.join(self.storage.path, self.base_model)
        state_dict = self.storage.load_file(model, "Checkpoint")
        state_dict = utils.cast_state_dict(state_dict, torch.float16)

        model_type, model_variant, prediction_type = "SDv1", "", "epsilon"

        unet = models.UNET(model_type, model_variant, prediction_type, dtype)
        unet.load_state_dict(state_dict["UNET"], strict=False)
        unet.to(device, dtype)
        unet.train(False)

        clip = models.CLIP(model_type, dtype)
        clip.model.load_state_dict(state_dict["CLIP"], strict=False)
        clip.to(device, dtype)
        clip.train(False)

        vae = models.VAE(model_type, dtype)
        vae.load_state_dict(state_dict["VAE"], strict=False)
        vae.to(device, torch.float32)

        del state_dict

        if self.folders:
            dataset = train.Dataset()
            for folder in self.folders:
                dataset.add_path(folder)
        elif self.dataset != None:
            dataset = self.dataset
        else:
            raise RuntimeError("No dataset specified")

        self.set_status("Preparing")

        dataset.configure(vae, clip, batch_size, image_size)
        dataset.calculate_buckets()
        dataset.build_cache(callback=self.on_caching_step)
        dataset.calculate_batches()

        del vae
        self.storage.do_gc()
        
        rank = int(self.lora_rank)
        conv_rank = int(self.lora_conv_rank)
        alpha = int(self.lora_alpha)

        network, modules = train.build_lora(rank, conv_rank, alpha, unet, clip)
        network = network.to(device, torch.float32)

        parameters = []
        for name, module in modules.items():
            parameters += [module.lora_module.lora_down.parameters()]
            parameters += [module.lora_module.lora_up.parameters()]
        parameters = itertools.chain(*parameters)
        network.train(True)

        epochs = dataset.total_epochs(steps)
        num_warmup_steps = int(self.warmup * steps)

        optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
        noise_scheduler = diffusers.DDPMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
        )

        if self.learning_schedule == "Cosine":
            lr_scheduler = diffusers.optimization.get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer, num_warmup_steps, steps, self.restarts
            )
        elif self.learning_schedule == "Linear":
            lr_scheduler = diffusers.optimization.get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps, steps
            )
        elif self.learning_schedule == "Constant":
            lr_scheduler = diffusers.optimization.get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps, steps
            )
        
        step = 0

        progress = tqdm.tqdm(total=steps, desc=f"TRAINING", disable=False)

        self.set_status("Training")
        loss_window = []
        loss_last = []
        
        last_time = 0
        for epoch in range(1, epochs+1):
            dataset.calculate_batches()
            for i, latents, tokens in dataset:
                with torch.no_grad():
                    latents = latents.to(device)
                    noise = torch.randn_like(latents, device=device, dtype=torch.float32)
                    timesteps = torch.randint(0, 1000, (batch_size,), device=device).long()
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    latents = latents.to(torch.device("cpu"))

                with torch.cuda.amp.autocast():
                    conditioning = torch.cat(train.encode_tokens(clip, tokens, clip_skip))
                    prediction = unet(noisy_latents, timesteps, encoder_hidden_states=conditioning).sample
                    loss = torch.nn.functional.mse_loss(prediction, noise, reduction="mean")

                loss.backward()

                torch.nn.utils.clip_grad_norm(parameters, 1.0)

                optimizer.step()
                lr_scheduler.step()

                optimizer.zero_grad(set_to_none=True)

                step += 1
                loss_window.insert(0, loss.item())
                if len(loss_window) > len(dataset):
                    loss_window.pop()
                current_loss = sum(loss_window)/len(loss_window)
                loss_last += [current_loss]
                
                progress.update(1)
                progress.set_postfix({"loss": current_loss})

                if time.time() - last_time > 1:
                    last_time = time.time()
                    self.on_training_step(progress.format_dict, len(dataset), loss_last)
                    loss_last = []

                if step % 1000 == 0 or step == steps:
                    self.set_status("Saving")
                    state_dict = train.get_state_dict(network)

                    filename = f"{self.name}_{step}.safetensors"
                    folder = os.path.join(self.storage.path, "LoRA", self.name)
                    os.makedirs(folder, exist_ok=True)

                    safetensors.torch.save_file(state_dict, os.path.join(folder, filename))
                    self.set_status("Training")
                
                if step == steps:
                    break

        self.storage.reset()

        if not self.callback({"type": "done", "data": {}}):
            raise RuntimeError("Aborted")
        
    def train_upload(self):
        if self.index == 0:
            self.dataset = train.Dataset()
        self.dataset.add_image(self.image[0], self.prompt[0])

        print(self.index)
        if not self.callback({"type": "training_upload", "data": {"index": self.index}}):
            raise RuntimeError("Aborted")