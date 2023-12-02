import argparse
import os
from dotenv import load_dotenv
import json
from io import BytesIO
import random
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel  # PNDMScheduler
from diffusers import LMSDiscreteScheduler
from tqdm.auto import tqdm

try:
    import swagger_server.services.config as config
except:
    import config

import pika
import openai
import requests
import torch
from PIL import Image

load_dotenv()

credentials = pika.PlainCredentials('admin', os.getenv("RABBIT_MQ_PW"))
virtual_host = config.CONFIG['rabbit_mq']['virtual_host']
host = config.CONFIG['rabbit_mq']['host']

model_id = "prompthero/openjourney"
device = "cuda" if torch.cuda.is_available() else "cpu"

if device != "cuda":
    print("WARNING: Not using CUDA. This will be very slow.")

openai.api_key = os.getenv("OPENAI_OPENAI_KEY")

first_connection = pika.BlockingConnection(
    pika.ConnectionParameters(host=host, credentials=credentials, virtual_host=virtual_host))

first_channel = first_connection.channel()

first_channel.exchange_declare(exchange='sender', exchange_type='topic')

gpt3_receiver_channel = first_connection.channel()
gpt3_receiver_channel.exchange_declare(exchange='sender', exchange_type='topic')
gpt3_receiver_result = gpt3_receiver_channel.queue_declare('', exclusive=True)
gpt3_queue = gpt3_receiver_result.method.queue
gpt3_receiver_channel.queue_bind(exchange='sender', queue=gpt3_queue, routing_key='gpt3_request')

audio_receiver_channel = first_connection.channel()
audio_receiver_channel.exchange_declare(exchange='sender', exchange_type='topic')
audio_receiver_result = audio_receiver_channel.queue_declare('', exclusive=True)
audio_queue = audio_receiver_result.method.queue
audio_receiver_channel.queue_bind(exchange='sender', queue=audio_queue, routing_key='audio_request')

stable_diffusion_receiver_channel = first_connection.channel()
stable_diffusion_receiver_channel.exchange_declare(exchange='sender', exchange_type='topic')
stable_diffusion_result = stable_diffusion_receiver_channel.queue_declare('', exclusive=True)
stable_diffusion_queue = stable_diffusion_result.method.queue
stable_diffusion_receiver_channel.queue_bind(exchange='sender', queue=stable_diffusion_queue,
                                             routing_key='stable_diffusion_request')


def bytes_to_json(body):
    return json.loads(body)


def second_sender_connection(response, routing_key):
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host=host, credentials=credentials, virtual_host=virtual_host)
    )
    second_sender_channel = connection.channel()
    second_sender_channel.exchange_declare(exchange='response', exchange_type='topic')
    second_sender_channel.basic_publish(
        exchange='response',
        routing_key=routing_key,
        body=response)
    connection.close()


def get_on_request_fn(model: str):
    if model == "gpt-3":
        handle_request_fn = handle_gpt3_request
        routing_key = "gpt3_response"

    elif model == "audio":
        handle_request_fn = handle_audio_request
        routing_key = "audio_response"

    elif model == "stable_diffusion":
        handle_request_fn = handle_stable_diffusion_request
        routing_key = "stable_diffusion_response"

    def on_request(ch, method, props, body):
        print("the model has been request - on_request")
        response = handle_request_fn(body)
        second_sender_connection(response, routing_key)

    return on_request


def handle_gpt3_request(body):
    print("handle_gpt3_request - rpc_server")
    body = bytes_to_json(body)
    prompt = body['prompt']
    prompt_enhanced_id = body['prompt_enhanced_id']

    try:
        enhanced_prompt = gpt3_model(prompt)

        response = {
            'status': "success",
            'prompt_enhanced_id': prompt_enhanced_id,
            'enhanced_prompt': enhanced_prompt,
            'error': "",
        }

    except Exception as e:
        response = {
            'status': "fail",
            'prompt_enhanced_id': prompt_enhanced_id,
            'enhanced_prompt': "",
            'error': str(e)
        }
    response = json.dumps(response)
    return response


def handle_audio_request(body):
    try:
        data = body[84:]
        body = bytes_to_json(body[:84])
        print("entered handle_audio_request")
        transcription_id = body['transcription_id']
        transcript_str, transcript_translated = get_transcription(data, body['audio_language'])
        response = {
            'status': "success",
            'transcription_id': transcription_id,
            'transcript_content': transcript_str,
            'transcript_content_translated': transcript_translated,
            'error': "",
        }

    except Exception as e:
        print(e)
        response = {
            'status': "fail",
            'error': str(e)
        }
    response = json.dumps(response)
    return response


def handle_stable_diffusion_request(body):
    body = bytes_to_json(body)
    try:
        images_array_id = body['images_array_id']
        images_array = generate_images(body['prompt'], body['height'], body['width'], images_array_id)

        response = {
            'status': "in progress",
            'images_array_id': images_array_id,
            'images_array': images_array,
            'error': "",
            'images_progress': 'done'
        }
    except Exception as e:
        response = {
            'status': "fail",
            'error': str(e)
        }
    response = json.dumps(response)
    second_sender_connection(response, 'stable_diffusion_response')

    return response


""""Models"""


def diffusers_pipeline(prompt_enhanced, num_inference_steps, height, width, images_array_id):
    prompt = prompt_enhanced
    guidance_scale = 7.5
    seed = random.randint(0, 1000000)
    generator = torch.manual_seed(seed)

    batch_size = 2

    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True,
                           return_tensors="pt")

    with torch.no_grad():

        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

        print(text_embeddings.shape)

        text_embeddings = text_embeddings.repeat(batch_size, 1, 1)

        print(text_embeddings.shape)

    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    latents = torch.randn(
        (batch_size, unet.in_channels, height // 8, width // 8),
        generator=generator,
    )
    latents = latents.to(device)
    scheduler.set_timesteps(num_inference_steps)
    latents = latents * scheduler.init_noise_sigma
    result_array = []
    counter = 0
    for t in tqdm(scheduler.timesteps):

        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample
        counter += 1

        if counter % 2 == 0 or counter == 1:
            with torch.no_grad():
                image = vae.decode((1 / 0.18215) * latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
            images = (image * 255).round().astype("uint8")
            pil_images = [Image.fromarray(image) for image in images]
            new_pil_images0 = pil_images[0].tobytes().decode('latin1')
            new_pil_images1 = pil_images[1].tobytes().decode('latin1')
            if len(result_array) == 0:
                result_array.append(pil_images[0].tobytes().decode('latin1'))
                result_array.append(pil_images[1].tobytes().decode('latin1'))
            else:
                result_array[0] = new_pil_images0
                result_array[1] = new_pil_images1

            response = {
                'status': "in progress",
                'images_array_id': images_array_id,
                'images_array': result_array,
                'error': "",
                'images_progress': 'in progress'
            }
            response = json.dumps(response)
            second_sender_connection(response, 'stable_diffusion_response')

    return result_array


def gpt3_model(prompt):
    gpt3_prompt = f"""I want you to act as a Open Journey Prompt Generator. A good prompt needs to be detailed and specific. A good process is to look through a list of keyword categories and decide whether you want to use any of them. The formula for a prompt is made of parts, the parts are indicated by brackets. The [Subject] is the person, place or thing the image is focused on. A common trick for human subjects is to use celebrity names. They have a strong effect and are an excellent way to control the subject’s appearance. However, be aware that these names may change not only the face but also the pose and something else. The [Medium] is the material used to make the generative artwork. Some examples are illustration, oil painting, 3D rendering, movie poster, sketch, digital art, portraits, cartoon, figure, scene, concept art, landscape, anime drawing, and photography. Medium has a strong effect because one keyword alone can dramatically change the style. The [Style] refers to the artistic style of the image. Examples include impressionist, cubism, surrealist, pop art, fantasy, art deco, art nouveau, 3d, baroque, pixar, isometric, graffiti art, and vtuber. The [Artist] is a strong modifier. It allows you to dial in the exact style using a particular artist as a reference. It is also common to use multiple artist names to blend their styles. Examples include vincent van gogh, pablo picaso, thomas kinkade, frida khalo, tooth wu, hr giger, ross tran, kim jung gi, russ mills, studio ghibli, and toei animations. Niche graphic [Websites] aggregates many images of distinct genres. Using them in a prompt is a sure way to steer the image toward these styles. Examples include artstation, Deviantart, pixabay, pixabay, pixiv, sephiroth art, and cgsociety. The [Resolution] represents how sharp and detailed the image is. Example of Resolution modifier includes 4k, 8k, 100m, Canon50, Fujifilm XT3, DSLR, ultra-highres, soft focus, cgi, ray tracing, arnold render, houdini render, octane render, and unreal engine. The addition of [Details] is a modifier to the image with the purpose of further refine and improve visuals. Examples include detailed, masterpiece, smooth, epic, angelic, photorealistic, award winning photo, fractal, sci-fi, stunningly beautiful, dystopian, intricate, and bokeh. The [Color] controls the overall visual of the image by adding different tones to the Subject or any other object. Examples include iridescent gold, silver, black, vivid, vintage, aesthetic, neon, muted colors, ektachrome, lush, and high contrast. The [Lighting] is a key factor in creating successful images, as Lighting can have a huge effect on how the image looks. The right balance of Lighting can produce the correct shadows and help the image by providing dimension, pulling focus, and strengthening contrast. Example includes cinematic lighting, dark, sunlight, cell shading, god rays, hard shadows, translucent, volumetric lighting, luminescence, long exposure, and bioluminescence. The prompt is then built using the [Medium] [Subject] [Style] [Artist] [Website] [Resolution] [Additional details] [Color] [Lighting]. I will give you a [Subject], you will respond with a full prompt. Present the result as one full sentence, no line breaks, no delimiters, and keep it as concise as possible while still conveying a full scene.

    I will give you some examples  for you to understand it better:
    The first line is the INPUT, is the Subject I send give you.
    The second line is the OUTPUT, the result i expect.
    
    1-)female elf
    portrait of female elf, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by artgerm and greg rutkowski and alphonse mucha, 8k
    
    2-)Japanese shrine on top of a mountain
    japanese style shrine on top of a misty mountain overgrown, hyper realistic, lush gnarly plants, 8 k, denoised, by greg rutkowski, tom bagshaw, james gurney cinematic lighting
    
    3-)cyberpunk bedroom
    cute isometric cyberpunk bedroom, cutaway box, futuristic, highly detailed, made with blender --v 4
    
    4-)Happy fox with sunglasses
    mdjrny-v4 style highly detailed matte painting stylized three quarters portrait of an anthropomorphic rugged happy fox with sunglasses! head animal person, background blur bokeh!  
    
    5-)cutest and softest creature in the world
    cutest AND softest creature in the world| large doll like eyes| supernatural and otherworldly| highly detailed vibrant fur| magical glowing trails| light dust| aesthetic| cinematic lighting| bokeh effect| mdjrny-v4 style
    
    6-)Multiple connecting tunnels in antartica bellow thin ice, multiple people moving around the tunnels
    Multiple connecting tunnels in antartica bellow thin ice, multiple people moving around the tunnels, The wind chimes of years ago floated in the still overcast day, a light leaking from my ears, buzzing, distant glaciers rushing towards me. facinating and imposing, fantasy digital art, octane render, beautiful composition, trending on artstation, award-winning photograph, masterpiece
    
    7-)gorgeous blonde female
    photo of a gorgeous blonde female in the style of stefan kostic, realistic, half body shot, sharp focus, 8 k high definition, insanely detailed, intricate, elegant, art by stanley lau and artgerm, extreme blur cherry blossoms background
    
    8-)fairytale treehouse village
    valley, fairytale treehouse village covered,  matte painting, highly detailed, dynamic lighting, cinematic, realism, realistic, photo real, sunset, detailed, high contrast, denoised, centered, michael whelan
    
    Subject: {prompt}
    Dont end with a .
    Dont start with a (dot)."""

    response = openai.Completion.create(
        model=config.CONFIG['openai']['gpt3_model_id'],
        prompt=gpt3_prompt,
        temperature=0.7,
        max_tokens=300,
        n=1
    )
    print('response')
    enhanced_prompt = str([f"{c['text'].strip()}" for idx, c in enumerate(response['choices'])][0])
    print('enhanced_prompt')
    print(enhanced_prompt)
    return enhanced_prompt


def translate_transcript(transcript):
    print("entered translate_transcript - rpc_server")
    res = requests.post(f'{config.CONFIG["audio"]["mt_host"]}/de_to_en',
                        data={'apikey': os.getenv("AUDIO_MT_APIKEY"), 'text': transcript})
    transcript_str = res.json()['text']
    return transcript_str


def get_transcription(data, audio_language):
    print("entered get_transcription")
    if audio_language == 'CH':
        audio_host = config.CONFIG['audio']['ch_audio_host']
        apikey = os.getenv("AUDIO_CH_AUDIO_APIKEY")

    else:
        audio_host = config.CONFIG['audio']['de_audio_host']
        apikey = os.getenv("AUDIO_DE_AUDIO_APIKEY")

    res = requests.post(
        audio_host,
        files={'audio': data},
        data={'apikey': apikey})

    transcript_str = res.json()['transcript_str']
    transcript_translated = translate_transcript(transcript_str)
    return transcript_str, transcript_translated


def pil_img_to_byte(pil_img: Image) -> str:
    """
    Convert a PIL image to a byte array.
    :param pil_img: PIL image
    :return: Byte array
    """
    pil_img.tobytes()
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG')
    return img_io.getvalue()


def generate_images(prompt_enhanced, height, width, images_array_id):
    print("Generating images function - rpc server")
    image = diffusers_pipeline(prompt_enhanced=prompt_enhanced,
                               num_inference_steps=50,
                               height=height, width=width, images_array_id=images_array_id)
    print("image generated")

    return image


"""Starting the server"""
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="gpt-3", choices=['gpt-3', 'audio', 'stable_diffusion'], type=str,
                    help="The model to start")
args = parser.parse_args()

if args.model == 'gpt-3':
    print("Starting Gpt-3 model")
    print('gpt3_receiver_channel consuming')
    gpt3_receiver_channel.basic_qos(prefetch_count=1)
    gpt3_receiver_channel.basic_consume(queue=gpt3_queue, auto_ack=True, on_message_callback=get_on_request_fn('gpt-3'))
    gpt3_receiver_channel.start_consuming()

elif args.model == 'audio':
    print("Starting Audio model")
    print('audio_receiver_channel consuming')
    audio_receiver_channel.basic_qos(prefetch_count=1)
    audio_receiver_channel.basic_consume(queue=audio_queue, auto_ack=True,
                                         on_message_callback=get_on_request_fn('audio'))
    audio_receiver_channel.start_consuming()

elif args.model == 'stable_diffusion':
    print("Loading Stable Diffusion model")

    # 1. Load the autoencoder model which will be used to decode the latents into image space.
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")

    # 2. Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

    # 3. The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")

    scheduler = LMSDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

    vae = vae.to(device)
    text_encoder = text_encoder.to(device)
    unet = unet.to(device)

    print('stable_diffusion_receiver_channel consuming')
    stable_diffusion_receiver_channel.basic_qos(prefetch_count=1)
    stable_diffusion_receiver_channel.basic_consume(queue=stable_diffusion_queue, auto_ack=True,
                                                    on_message_callback=get_on_request_fn('stable_diffusion'))
    stable_diffusion_receiver_channel.start_consuming()
