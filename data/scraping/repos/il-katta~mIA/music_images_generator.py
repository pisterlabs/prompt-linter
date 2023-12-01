import json
from typing import List, Tuple, Any

import openai
import gradio as gr

from utils import package_exists, cuda_is_available
from utils.system_stats import SystemStats


def is_available():
    return package_exists("openai") \
        and package_exists("transformers") \
        and package_exists("torch") \
        and package_exists("diffusers") \
        and cuda_is_available()


def call_openai_api(text: str):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        # model="gpt-4-0613",
        messages=[
            {
                "role": "user",
                "content": f"Generate image for the song {text}"
            }
        ],
        functions=[
            {
                "name": "generate_image",
                "description": """Generate an image based on the description of the elements present in the image and the type of image you want to generate.
the description of the image are a list of details that must be present in the image representative of the song lyrics or the song meaning. 
The details must be concrete, devoid of abstract concepts, for example instead of 'A corrupted city' the correct parameter is 'A city, garbage on the street, burning cars'. 
The image ca be realistic or fantasy, you will also need to specify if the image is a photo therefore a real thing or a drawing because it includes elements of fantasy.
""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subject": {
                            "type": "string",
                            "description": "the description of the elements present in the image, separated by comma. For example: 'a man, a dog, a tree, night, moon'",
                        },
                        "type": {
                            "type": "string",
                            "enum": ["realistic", "fantasy"],
                            "description": "the type of image you want to generate, 'realistic' if the image should be realistic, 'fantasy' if the image should be fantasy"
                        }
                    },
                    "required": ["subject", "type"]
                }
            }
        ],
        temperature=1,
    )
    for choice in response.choices:
        if choice.message.get("function_call", None):
            function_name = choice.message["function_call"]["name"]
            args = json.loads(choice.message["function_call"]["arguments"])
            yield function_name, args
        try:
            args = json.loads(choice.message["content"])
            yield "generate_image", args
        except:
            pass


def clean_all():
    return "", ""


def gui(sysstats: SystemStats):
    from utils.image_generator import ImageGenerator
    from utils.prompt_generator import PromptGenerator

    image_generator = ImageGenerator(model_name="rundiffusionFX")
    sysstats.register_disposable_model(image_generator)

    prompt_generator = PromptGenerator()
    sysstats.register_disposable_model(prompt_generator)

    def on_input_song(song_artist_title: str, images: List[Tuple[str, Any]], only_text: bool = False):
        '''

        :param song_artist_title:
        :return: message_textbox, prompt_markdown, image_viewer, image_gallery, images
        '''

        for txt in call_openai_api(song_artist_title):
            subject = txt[1]["subject"]
            yield f"🎵 {song_artist_title}", f"*{subject}*", None, images, images

            expanded_subject = prompt_generator.generate_prompt(subject)

            # subject_markdown = f"_{subject}_\n\n**{expanded_subject}**"
            subject_markdown = expanded_subject.replace("subject", f"**subject**")

            yield f"🎵 {song_artist_title}", subject_markdown, None, images, images

            if only_text:
                import time
                time.sleep(10)
                continue

            for i in range(5):
                image, metadata = image_generator.generate_image(expanded_subject, sampler_name="DPM Solver++")
                images.append((image, song_artist_title))
                yield f"🎵 {song_artist_title}", subject_markdown, image, images, images

    images = gr.State([])

    with gr.Row():
        with gr.Column(scale=5, min_width=100):
            message_textbox = gr.Textbox(
                show_label=False,
                placeholder="Enter song title and artist",
                container=False,
                lines=1,
            )

        with gr.Column(scale=1, min_width=20):
            submit_button = gr.Button("↗️", variant="primary")
            clear_button = gr.Button("🗑", variant="secondary")

    with gr.Row():
        generate_only_text_checkbox = gr.Checkbox(label="📝 Generate only prompt", value=False)

    with gr.Row():
        song_markdown = gr.Markdown("")

    with gr.Row():
        prompt_markdown = gr.Markdown("")

    with gr.Row():
        image_viewer = gr.Image(type="pil", image_mode="RGB")

    with gr.Row():
        image_gallery = gr.Gallery(
            label="Generated images", show_label=True, elem_id="gallery", value=images.value,
        )

    clear_button.click(
        clean_all,
        inputs=[],
        outputs=[message_textbox, prompt_markdown],
        queue=False
    )

    submit_button.click(
        lambda x: x,
        inputs=[message_textbox],
        outputs=[song_markdown],
        queue=False,
        api_name="_",
    ).then(
        on_input_song,
        inputs=[message_textbox, images, generate_only_text_checkbox],
        outputs=[song_markdown, prompt_markdown, image_viewer, image_gallery, images],
        api_name="generate_image_from_song",
    )
