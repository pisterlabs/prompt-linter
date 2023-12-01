from pathlib import Path
import openai
import requests

from CreatorPipeline.constants import PLATFORMS, PROMPTS

openai.api_key = PLATFORMS.openai.get("OPENAI_KEY")


def generate_text(prompt, output=None):
    """Generates text from a prompt using the OpenAI API."""
    response = openai.Completion.create(
        model="text-davinci-003", prompt=prompt, max_tokens=2000
    )
    if not output:
        return response.choices[0].text
    with open(output, "w") as f:
        f.write(response.choices[0].text)


def generate_image(prompt, n, output=None):
    """Generates images from a prompt using the OpenAI API."""
    response = openai.Image.create(prompt=prompt, n=n, size="1024x1024")
    image_urls = response["data"][0]["url"]

    if not output:
        return image_urls

    for i, pic in enumerate(image_urls):
        pic_path = Path(f"{output}") / f"thumbnail_component_{i:04d}.png"
        save_image(pic_url=pic, output=pic_path)


def transcribe_audio(fp, output=None):
    """Transcribes audio from a file using the OpenAI API."""
    print(help(openai.Audio.transcribe))
    with open(fp, "rb") as f:
        response = openai.Audio.transcribe(model="whisper-1", file=f)
        if response:
            print(response)
            return response


def translate_audio(file, lang="English", output=None):
    """Translates audio from a file using the OpenAI API."""
    openai.Audio.translate(model="whisper-1", file=file, language=lang)


def save_image(pic_url, output):
    """Saves an image from a url to a file."""
    with open(output, "wb") as handle:
        response = requests.get(pic_url, stream=True)

        if not response.ok:
            print(response)

        for block in response.iter_content(1024):
            if not block:
                break
            handle.write(block)


# received from db
params = {
    "SUBJECT": "Landing Your First Job",
    "GAIN": "Know how to confidently get a first job opportunity.",
}
params = {
    "SUBJECT": "",
    "GAIN": "",
    "TITLE": "",
    "RELEASE_DATE": "",
    "VIDEO1": "",  # choice from options
    "VIDEO2": "",  # choice from options
}


# Define might need a step first to set the gain desired?
# or maybe the clarity can be suggested?
# maybe its just manual
def generate_episode_ai_content(params, episode_root=None):
    """Generates AI content for an episode."""
    print("\nGenerating AI Content")
    episode_root = Path(episode_root)

    # to 03_script/script.txt
    generate_text(
        PROMPTS.script_outline.substitute(**params),
        episode_root / "03_script/script.txt",
    )
    # to 02_research/components/thumb
    generate_image(
        PROMPTS.thumbnail_components.substitute(**params),
        PROMPTS.thumbnail_component_count,
        episode_root / "02_research/components/thumb",
    )
    # to 07_release/community
    generate_text(
        PROMPTS.community.substitute(**params),
        episode_root / "07_release/community/community.txt",
    )
    # to 07_release/emails
    generate_text(
        PROMPTS.email_chain.substitute(**params),
        episode_root / "07_release/emails/email_pre.txt",
    )
    # to 08_market/socials
    generate_text(
        PROMPTS.channel_article.substitute(**params),
        episode_root / "08_market/socials/article.txt",
    )
    # to 08_market/emails
    generate_text(
        PROMPTS.blog_post.substitute(**params),
        episode_root / "08_market/emails/emails_post.txt",
    )
    print("\nDone Generating AI Content\n")


def generate_closed_captions(audio_file, output=None):
    """Generates closed captions for an audio file."""
    transcribe_audio(audio_file, output)
    # for lang in PROMPTS.languages:
    #     translate_audio(audio_file, lang, output)

