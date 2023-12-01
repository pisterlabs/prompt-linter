import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
import traceback

from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from torch import autocast

torch.set_grad_enabled(False)

import openai
import asyncio
import time
import openai.error
openai.api_key = "YOUR_API_KEY"


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

with open('checkpoint.txt') as reader:
    CKPT = "\n".join(reader.readlines())

async def get_des(text, instruction):
    completion = openai.ChatCompletion.create(engine="gpt-3.5-turbo", messages=[
    {"role": "user", "content":
f'''Instruction:
{CKPT}

Happy generating!

Input Caption: {text}
Concept: {instruction}
Output Caption: '''},
    ])

    content = completion["choices"][0]["message"]["content"]
    return content

async def get_tgt(text, instruction):
    while True:
        try:
            task = asyncio.create_task(get_des(text, instruction))
            await asyncio.wait_for(task, timeout=10)
            content = task.result()
        except openai.error.RateLimitError:
            task.cancel()
            print('Rate Limit, wait 3s & retry')
            time.sleep(3)
            continue
        except asyncio.TimeoutError:
            task.cancel()
            print('Timeout, retry')
            continue
        except:
            task.cancel()
            print('Unkown error, retry')
            print(traceback.format_exc())
            continue
        else:
            break
    return content

def load_model_from_config(config, ckpt, device=torch.device("cuda"), verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    if device == torch.device("cuda"):
        model.cuda()
    elif device == torch.device("cpu"):
        model.cpu()
        model.cond_stage_model.device = "cpu"
    else:
        raise ValueError(f"Incorrect device name. Received: {device}")
    model.eval()
    return model

config = OmegaConf.load("configs/stable-diffusion/v2-inference-v.yaml")
device = torch.device("cuda")
model = load_model_from_config(config, "checkpoints/v2-1_768-ema-pruned.ckpt", device)
ddim_sampler = DDIMSampler(model, device)


def process(width, height, prompt, instruction, a_prompt, n_prompt, num_samples, ddim_steps, guess_mode, strength, scale, seed, eta):
    with torch.no_grad():

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        neg_cap = asyncio.run(get_tgt(prompt, instruction))
        print("Raw: ", prompt)
        print("Neg: ", neg_cap)

        ddim_sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=eta, verbose=False)

        precision_scope = autocast
        with torch.no_grad(), precision_scope("cuda"), model.ema_scope():
            cond = model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)
            cond2 = model.get_learned_conditioning([neg_cap + ', ' + a_prompt] * num_samples)
            un_cond = model.get_learned_conditioning([n_prompt] * num_samples)
            shape = (4, width // 8, height // 8)

            model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                         shape, cond, cond2, verbose=False, eta=eta,
                                                         unconditional_guidance_scale=scale,
                                                         unconditional_conditioning=un_cond, strength=strength)
                                                         #x_T=start_code)

            x_samples = model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return results


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## LTF")
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="User's Query", value="A Selfy of SpaceX's Founder")
            instruction = gr.Textbox(label="Adiministator's Concept", value="Elon Musk")
            strength = gr.Slider(label="Ratio ( S = ratio * T )", minimum=0.0, maximum=2.0, value=0.2, step=0.01)
            run_button = gr.Button(label="Run")
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                width = gr.Slider(label="Image Resolution of Width", minimum=256, maximum=1024, value=768, step=64)
                height = gr.Slider(label="Image Resolution of Height", minimum=256, maximum=1024, value=768, step=64)
                guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                eta = gr.Number(label="eta (DDIM)", value=0.0)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                n_prompt = gr.Textbox(label="Negative Prompt",
                                      value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
    ips = [width, height, prompt, instruction, a_prompt, n_prompt, num_samples, ddim_steps, guess_mode, strength, scale, seed, eta]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery])


block.launch(server_name='0.0.0.0')
