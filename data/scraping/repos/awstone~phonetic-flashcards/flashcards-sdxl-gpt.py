import gradio as gr
import pathlib
import numpy as np
import cv2
import random
from hf_hub_ctranslate2 import TranslatorCT2fromHfHub, GeneratorCT2fromHfHub
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from diffusers import DiffusionPipeline
import math
from PIL import Image, ImageFont, ImageDraw

import openai
import os
from dotenv import load_dotenv
load_dotenv('/home/awstone/.bashrc')
openai.api_key = os.environ["OPENAI_API_KEY"]


# load both base & refiner
base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
        ).to("cuda")
refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        ).to("cuda")


def call_model(user_prompt, art_style, num_imgs):   
    
    with open('system-prompt.txt', 'r') as file:
        system_prompt = file.read()
    with open('prompt1.txt', 'r') as file:
        prompt1 = file.read()
    with open('prompt2.txt', 'r') as file:
        prompt2 = file.read()
    prompt2 += user_prompt
    with open('prompt3.txt', 'r') as file:
        prompt3 = file.read()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt1},
    ]

    response = openai.ChatCompletion.create(
    model="gpt-4",
    # model="gpt-4",
    messages=messages
    )
    print("response 1: ", response)

    messages.append({"role": "assistant", "content": response['choices'][0]['message']['content']})
    messages.append({"role": "user", "content": prompt2})
    response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=messages
    )
    print('response 2: ', response)

    messages.append({"role": "assistant", "content": response['choices'][0]['message']['content']})
    messages.append({"role": "user", "content": prompt3})
    response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=messages
    )
    print('response 3: ', response)

    csv = response['choices'][0]['message']['content']
    objects = csv.split('\n')[-1].split(',')
    new_objects = []
    for i in range(0, len(objects), 2):
        new_objects.append(objects[i] + ' ' + objects[i+1])
    objects = new_objects
    print("Objects are: ", objects)
    for i, obj in enumerate(objects):
        if i < int(num_imgs):   
            prompt = f"a {art_style} of a {obj.strip().split(' ')[0]}"
            # Define how many steps and what % of steps to be run on each experts (80/20) here
            n_steps = 40
            high_noise_frac = 0.8

            # run both experts
            image = base(
                prompt=prompt,
                num_inference_steps=n_steps,
                # denoising_end=high_noise_frac,
                output_type="latent",
            ).images
            image = refiner(
                prompt=prompt,
                num_inference_steps=n_steps,
                # denoising_start=high_noise_frac,
                image=image,
            ).images[0]
            image.save(f"images/{obj.strip().split(' ')[0]}.png")
        else:
            break
    return objects[0:int(num_imgs)]

def get_text_dimensions(text_string, font):
    # https://stackoverflow.com/a/46220683/9263761
    ascent, descent = font.getmetrics()

    text_width = font.getmask(text_string).getbbox()[2]
    text_height = font.getmask(text_string).getbbox()[3] + descent

    return (text_width, text_height)

def add_text_to_image(image_path, text, x, y, font_size=1.0, font_color=(0, 0, 0), thickness=2):
    # Read the image
    image = image_path

    # Define the font properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = font_size
    font_thickness = thickness
    # Calculate the size of the text to get its width and height
    # (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype('/home/awstone/dejavu-fonts-ttf-2.37/ttf/DejaVuSerif.ttf', size=30, encoding='unic')
    text_width, text_height = get_text_dimensions(text, font)

    # Calculate the coordinates to center the text at the specified (x, y) point
    text_x = x - text_width // 2
    text_y = y - text_height // 2

    # Draw the text on the image
    # cv2.putText(image, text, (text_x, text_y), font, font_scale, font_color, font_thickness)

    
    
    # draw.text((text_x, text_y), text, font=font)
    print(f'text to go on card {text}')
    draw.text((text_x, text_y), text, font=font, fill='black')
    image = np.asarray(pil_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image

def overlay_images(base_image_path, overlay_image_path, x1, y1, x2, y2, name):
    # Read the base and overlay images
    base_image = cv2.imread(base_image_path)
    
    overlay_image = overlay_image_path

    # Get the dimensions of the overlay image
    overlay_height, overlay_width = overlay_image.shape[:2]

    # Calculate the width and height of the region to overlay
    width = x2 - x1
    height = y2 - y1

    # Resize the overlay image to match the size of the region to overlay
    resized_overlay = cv2.resize(overlay_image, (width, height))

    # Use the cv2.addWeighted function to blend the overlay image onto the base image
    alpha = 0  # Change this value to adjust the transparency of the overlay
    blended_image = cv2.addWeighted(base_image[y1:y2, x1:x2], alpha, resized_overlay, 1 - alpha, 0)

    # Replace the region of interest on the base image with the blended image
    base_image[y1:y2, x1:x2] = blended_image

    image_path = base_image
    text = name
    x_coordinate = 208
    y_coordinate = 450

    result_image = add_text_to_image(image_path, text, x_coordinate, y_coordinate)

    return result_image

def create_image_grid(images):
    num_images = len(images)
    # Calculate the number of rows and columns for the grid
    rows = int(math.sqrt(num_images))
    cols = int(math.ceil(num_images / rows))
    
    # Resize all images to have the same dimensions (assuming they are already of the same size)
    # max_height = max(img.shape[0] for img in images)
    # max_width = max(img.shape[1] for img in images)
    max_height = 800
    max_width = 600
    images = [cv2.resize(img, (max_width, max_height)) for img in images]
    
    # Create the grid by concatenating the images
    grid_rows = [cv2.hconcat(images[i:i+cols]) for i in range(0, num_images, cols)]
    grid_image = cv2.vconcat(grid_rows)

    return grid_image


def predict_objects(user_prompt,art_style,num_imgs):
    print('Prompt is: ', user_prompt)
    list_of_images = call_model(user_prompt, art_style, num_imgs)
    print("Objects to be displayed", list_of_images)
    output_list = []
    for i in range(len(list_of_images)):
        image_name = list_of_images[i]
        
        tempimg = cv2.imread("images/" + image_name.strip().split(' ')[0] + ".png")
        print("reading", "images/" + image_name.strip().split(' ')[0] + ".png")
        overlay_image_path = cv2.resize(tempimg, (tempimg.shape[0], tempimg.shape[0]))
        # overlay_image_path = make_image_circular(overlay_image_path)
        
        img_1 = np.zeros([60,140,1],dtype=np.uint8)
        img_1.fill(255)
        templatedirectory = "templates/"
        random_idx = random.randint(1, 8)

        # Example usage
        base_image_path = templatedirectory + 'template_' + str(random_idx) +'.png'
        x1, y1 = 90, 100
        x2, y2 = x1 + 235, y1 + 208  # Calculate the second coordinate based on width and height

        result_image = overlay_images(base_image_path, overlay_image_path, x1, y1, x2, y2, image_name)
        resized_image = cv2.resize(result_image, (415, 550))
        color_flipped_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        output_list.append(color_flipped_image)
    output = create_image_grid(output_list)
    return output

title = "Flashcards Demo For AI Institute"
description = "Demo for AI Institute"
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2203.02378' target='_blank'>Paper</a> | <a href='https://github.com/microsoft/unilm/tree/master/dit' target='_blank'>Github Repo</a></p> | <a href='https://huggingface.co/docs/transformers/master/en/model_doc/dit' target='_blank'>HuggingFace doc</a></p>"
examples =[['publaynet_example.jpeg']]
css = ".output-image, .input-image, .image-preview {height: 600px !important}"

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            user_prompt = gr.Textbox(label='user prompt', value="Generate a pair of nouns. One should have medial sound /r/. The other should have a minimally opposing phoneme to /r/. ")
            art_style = gr.Textbox(label='art style', value="minimal, transparent background, fun")
            num_imgs = gr.Textbox(label='num images', value="2")
            gen_story_btn = gr.Button('Generate Flashcards')
        with gr.Column():
            image = gr.Image()
    gen_story_btn.click(fn=predict_objects, inputs=[user_prompt,art_style,num_imgs], outputs=image)


demo.launch(share=True)

