import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from openai_util.function_call.openaifunc_decorator import openai_func
import os
import time

model_id = "stabilityai/stable-diffusion-2-1"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
try:
    pipe = pipe.to("cuda")
except Exception as e:
    print(e)
tmp_dir = os.path.join(os.getcwd(), "tmp")

@openai_func
def text_to_image(text: str):
    """
    generate image from english text
    :param text: the text must use english, for example: "a phone of a dog"
    """
    image = pipe(text).images[0]
    image_path = os.path.join(tmp_dir, f"{str(int(time.time()))}.png")
    image.save(image_path)
    return image_path

# text_to_image("a phone of a dog")
