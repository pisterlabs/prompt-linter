from typing import Optional
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

import base64
import io

import openai
import requests

import os

stage = os.environ.get("STAGE", None)
if stage is None:
    from climage.__main__ import _get_color_type, _toAnsi
    from PIL import Image

import gptif.settings
from gptif.console import console
from gptif.db import (
    AiImage,
    get_ai_image_from_id,
    get_ai_image_if_cached,
    put_ai_image_in_cache,
)


def display_image(image_data_bytes: bytes):
    if stage is None:
        im = Image.open(io.BytesIO(image_data_bytes))
        ctype = _get_color_type(
            is_truecolor=False, is_256color=True, is_16color=False, is_8color=False
        )
        output = _toAnsi(
            im, oWidth=80, is_unicode=True, color_type=ctype, palette="default"
        )
        print(output)


def _generate_image_openai(prompt: str) -> Optional[bytes]:
    try:
        # Grab the ai_image from openai
        import openai

        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size="512x512",
            response_format="b64_json",
        )

        image_data_b64 = response["data"][0]["b64_json"]
        image_data_bytes = base64.b64decode(image_data_b64)
        return image_data_bytes
    except Exception as ex:
        console.debug(ex)
        return None


def _generate_image_stability(prompt: str) -> Optional[bytes]:
    try:
        from stability_sdk import client
        import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

        for engine in [
            "stable-diffusion-xl-beta-v2-2-2",
        ]:
            # Set up our connection to the API.
            stability_api = client.StabilityInference(
                key=os.environ["STABILITY_KEY"],  # API Key reference.
                verbose=True,  # Print debug messages.
                engine=engine,  # Set the engine to use for generation.
                # Available engines: stable-diffusion-v1 stable-diffusion-v1-5 stable-diffusion-512-v2-0 stable-diffusion-768-v2-0
                # stable-diffusion-512-v2-1 stable-diffusion-768-v2-1 stable-diffusion-xl-beta-v2-2-2 stable-inpainting-v1-0 stable-inpainting-512-v2-0
            )

            # Set up our initial generation parameters.
            answers = stability_api.generate(
                prompt=prompt,
                seed=992446758,  # If a seed is provided, the resulting generated image will be deterministic.
                # What this means is that as long as all generation parameters remain the same, you can always recall the same image simply by generating it again.
                # Note: This isn't quite the case for CLIP Guided generations, which we tackle in the CLIP Guidance documentation.
                steps=30,  # Amount of inference steps performed on image generation. Defaults to 30.
                cfg_scale=7.0,  # Influences how strongly your generation is guided to match your prompt.
                # Setting this value higher increases the strength in which it tries to match your prompt.
                # Defaults to 7.0 if not specified.
                width=512,  # Generation width, defaults to 512 if not included.
                height=512,  # Generation height, defaults to 512 if not included.
                samples=1,  # Number of images to generate, defaults to 1 if not included.
                sampler=generation.SAMPLER_K_DPMPP_2M  # Choose which sampler we want to denoise our generation with.
                # Defaults to k_dpmpp_2m if not specified. Clip Guidance only supports ancestral samplers.
                # (Available Samplers: ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_dpmpp_2s_ancestral, k_lms, k_dpmpp_2m, k_dpmpp_sde)
            )

            # Set up our warning to print to the console if the adult content classifier is tripped.
            # If adult content classifier is not tripped, save generated images.
            for resp in answers:
                for artifact in resp.artifacts:
                    if artifact.finish_reason == generation.FILTER:
                        console.debug(
                            "Your request activated the API's safety filters and could not be processed."
                            "Please modify the prompt and try again."
                        )
                    elif artifact.type == generation.ARTIFACT_IMAGE:
                        return artifact.binary
        return None
    except Exception as ex:
        print("ERROR:")
        print(ex)
        console.debug(ex)
        return None


def _generate_image_sd_hf(prompt: str) -> Optional[bytes]:
    import requests

    API_URL = (
        "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
    )
    HF_KEY = os.environ["HUGGING_FACE_KEY"]
    headers = {"Authorization": f"Bearer {HF_KEY}"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.content

    image_bytes = query(
        {
            "inputs": prompt,
        }
    )

    return image_bytes


def generate_image(query: AiImage) -> Optional[bytes]:
    prompt = query.prompt
    print("GENERATE IMAGE WITH", query.model_version)
    print(prompt)

    if query.model_version == "dalle_with_waterfall":
        generators = [
            _generate_image_openai,
            _generate_image_stability,
            _generate_image_sd_hf,
        ]
        for g in generators:
            result = g(prompt)
            if result is not None:
                return result
        return None
    elif query.model_version == "dalle":
        return _generate_image_openai(prompt)
    elif query.model_version == "stability_ai":
        return _generate_image_stability(prompt)
    elif query.model_version == "stablediffusion_hf":
        return _generate_image_sd_hf(prompt)
    else:
        raise NotImplementedError(f"Invalid model type: {query.model_version}")


def display_image_for_prompt(prompt: str):
    print("DISPLAYING IMAGE FOR PROMPT", prompt)
    # if gptif.settings.DEBUG_MODE == True:
    # return
    query = AiImage(model_version="dalle_with_waterfall", prompt=prompt)
    if gptif.settings.CONVERSE_SERVER is None:
        ai_image = get_ai_image_if_cached(query)
        if ai_image is None:
            image_data_bytes = generate_image(query)
            query.result = image_data_bytes
            put_ai_image_in_cache(query)

            assert query.id is not None

            ai_image_id = query.id
        else:
            assert ai_image.id is not None
            ai_image_id = ai_image.id

        if ai_image_id is None:
            return
        if not gptif.settings.CLI_MODE:
            console.print(f"%%IMAGE%% {ai_image_id}")
            return

        ai_image = get_ai_image_from_id(ai_image_id)

        assert ai_image is not None
        if ai_image.result is not None:
            display_image(ai_image.result)

    else:
        response = requests.post(
            f"{gptif.settings.CONVERSE_SERVER}/fetch_image_id_for_caption",
            json=query.dict(),
        )

        console.debug("RESPONSE", response)
        console.debug(response.content)

        # TODO: More gracefully handle errors
        assert response.status_code == 200

        if response.content is not None:
            image_id = int(response.content.decode().strip('"'))

            response = requests.get(
                f"{gptif.settings.CONVERSE_SERVER}/ai_image/{image_id}"
            )

            # TODO: More gracefully handle errors
            assert response.status_code == 200

            # print(response.content)
            if not gptif.settings.CLI_MODE:
                console.print(f"%%IMAGE%% {image_id}")
                return

            # image_data_b64 = response["data"][0]["b64_json"]
            # image_data_bytes = base64.b64decode(image_data_b64)

            display_image(response.content)


if __name__ == "__main__":
    gptif.settings.CONVERSE_SERVER = "http://localhost:8000"
    display_image_for_prompt("Two dogs farting")
