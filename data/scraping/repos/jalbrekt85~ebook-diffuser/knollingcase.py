from ebook_difusser import EBookDiffuser
import os
from PIL import Image
import openai
from dotenv import load_dotenv


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class Knollingcase(EBookDiffuser):
    # configured for 8.25x11 hardcover Amazon books

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_profile()

    def generate_theme(self) -> str:
        res = openai.Completion.create(
            model=f"text-davinci-003",
            prompt=self.story.gpt_theme_prompt,
            temperature=1.0,
            max_tokens=15,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["\n"],
        )

        text = res["choices"][0]["text"][1:]
        print("theme: ", text)
        if text not in os.listdir(self.books_dir):
            return text

        self.generate_theme()

    def generate_page_prompt(self, theme) -> str:
        prompt = self.story.gpt_page_prompt.format(theme)
        res = openai.Completion.create(
            model=f"text-davinci-003",
            prompt=prompt,
            temperature=1,
            max_tokens=9,
            top_p=1,
            frequency_penalty=0.02,
            presence_penalty=0.02,
            stop=["\ntheme"],
        )

        page_prompt = res["choices"][0]["text"].split(":")[1][1:]

        # add latest result to gpt prompt template to avoid repetitive results
        self.story.gpt_page_prompt = prompt + "\nresponse: " + page_prompt + "\n" + "theme: {}"

        return "{} {}".format(theme, page_prompt)

    def generate_page_image(self, prompt) -> Image:
        res = self.api.txt2img(
            prompt=self.sd.prompt_template.format(prompt),
            negative_prompt=self.sd.negative_prompt,
            steps=self.sd.steps,
            cfg_scale=self.sd.cfg_scale,
            sampler_name=self.sd.sampler,
            width=self.sd.width,
            height=self.sd.height,
        )

        upscaled = self.api.extra_single_image(
            res.image, upscaler_1="ESRGAN_4x", upscaling_resize=3
        )

        return upscaled.image
