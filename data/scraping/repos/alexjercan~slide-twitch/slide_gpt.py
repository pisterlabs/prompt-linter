""" Slide Generation using GPT-4 and FakeYou """
import concurrent.futures
import glob
import json
import logging
import os
import urllib.request
import wave
from typing import Dict, Optional

import fakeyou
import ffmpeg
import openai
from dotenv import load_dotenv
from rich.progress import Progress

load_dotenv()

logger = logging.getLogger("slide_twitch")

SYSTEM = """Your job is to create a slide presentation for a video. \
In this presentation you must include a speech for the current slide and a \
description for the background image. You need to make it as story-like as \
possible. The format of the output must be in JSON. You have to output a list \
of objects. Each object will contain a key for the speech called "text" and a \
key for the image description called "image".

For example for a slide presentation about the new iphone you could output \
something like:

```
[
  {
    "text": "Hello. Today we will discuss about the new iphone",
    "image": "Image of a phone on a business desk with a black background"
  },
  {
    "text": "Apple is going to release this new iphone this summer",
    "image": "A group of happy people with phones in their hand"
  },
  {
    "text": "Thank you for watching my presentation",
    "image": "A thank you message on white background"
  }
]
```

Make sure to output only JSON text. Do not output any extra comments.
"""
SPEAKER = "TM:cpwrmn5kwh97"
MODEL = "gpt-4"

FAKEYOU_USERNAME = os.environ["FAKEYOU_USERNAME"]
FAKEYOU_PASSWORD = os.environ["FAKEYOU_PASSWORD"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


def create_video(output: str = os.path.curdir):
    """Create the video from the slides

    The video will be saved in the output directory as `video.mp4`. The video
    will be created by concatenating the images and audio files together.

    Parameters
    ----------
    output : str
        The output directory to use for the files, defaults to os.path.curdir

    Raises
    ------
    ValueError
        If the number of image and audio files is not the same
    Exception
        If anything else goes wrong
    """
    logger.debug("Creating video...")

    image_files = sorted(glob.glob(os.path.join(output, "slide_*.png")))
    audio_files = sorted(glob.glob(os.path.join(output, "slide_*.wav")))

    if len(image_files) != len(audio_files):
        raise ValueError("Number of image and audio files must be the same")

    input_streams = []
    for image_file, audio_file in zip(image_files, audio_files):
        input_streams.append(ffmpeg.input(image_file))
        input_streams.append(ffmpeg.input(audio_file))

    ffmpeg.concat(*input_streams, v=1, a=1).output(
        os.path.join(output, "video.tmp.mp4"),
        pix_fmt="yuv420p",
    ).overwrite_output().run()

    os.rename(
        os.path.join(output, "video.tmp.mp4"),
        os.path.join(output, "video.mp4"),
    )

    logger.debug("Video done")


def vtt_seconds_to_hh_mm_ss_mmm(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm format

    Parameters
    ----------
    seconds : float
        The seconds to convert

    Returns
    -------
    str
        The seconds in HH:MM:SS.mmm format
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    r_seconds = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)

    result = f"{hours:02d}:{minutes:02d}:{r_seconds:02d}.{milliseconds:03d}"

    return result


def create_vtt(output: str = os.path.curdir):
    """Create the VTT file for the presentation

    The SRT file will be saved in the output directory as `video.vtt`.
    The timing for each slide will be based on the `.wav` length.

    Parameters
    ----------
    output : str
        The output directory to use for the files, defaults to os.path.curdir

    Raises
    ------
    FileNotFoundError
        If the presentation file does not exist
    ValueError
        If the number of slides and audio files is not the same
    OSError
        If the presentation file cannot be opened
    Exception
        If anything else goes wrong
    """
    logger.debug("Creating vtt...")

    audio_files = sorted(glob.glob(os.path.join(output, "slide_*.wav")))

    with open(
        os.path.join(output, "presentation.json"), "r", encoding="utf-8"
    ) as file:
        presentation = json.load(file)

    if len(presentation) != len(audio_files):
        raise ValueError("Number of slides and audio files must be same")

    with open(
        os.path.join(output, "video.vtt"), "w", encoding="utf-8"
    ) as file:
        current_s = 0

        file.write("WEBVTT\n\n")

        for index, (slide, audio_file) in enumerate(
            zip(presentation, audio_files)
        ):
            with open(audio_file, "rb") as audio_f:
                audio = wave.open(audio_f)
                duration = audio.getnframes() / audio.getframerate()

            start = current_s
            end = current_s + duration

            start_fmt = vtt_seconds_to_hh_mm_ss_mmm(start)
            end_fmt = vtt_seconds_to_hh_mm_ss_mmm(end)

            file.write(f"{index + 1}\n")
            file.write(f"{start_fmt} --> {end_fmt}\n")
            file.write(f"{slide['text']}\n")
            file.write("\n")

            current_s = end

    logger.debug("VTT done")


def create_script(prompt: str, output: str = os.path.curdir) -> Dict:
    """Create the script for the presentation

    The script will be saved in the output directory as `presentation.json`.
    The script will be created by using the system prompt (from config) and
    the user prompt.

    Parameters
    ----------
    prompt : str
        The user prompt to use
    output : str
        The output directory to use for the files, defaults to os.path.curdir

    Returns
    -------
    Dict
        The presentation script

    Raises
    ------
    IndexError
        If the response is empty
    Exception
        If anything else goes wrong
    """
    logger.debug("Creating script for '%s'...", prompt)

    with open(
        os.path.join(output, "prompt.txt"), "w", encoding="utf-8"
    ) as file:
        file.write(prompt)

    response = openai.ChatCompletion.create(
        api_key=OPENAI_API_KEY,
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM,
            },
            {"role": "user", "content": prompt},
        ],
    )

    presentation = json.loads(response.choices[0].message.content)
    for slide in presentation:
        slide["speaker"] = SPEAKER

    with open(
        os.path.join(output, "presentation.json"), "w", encoding="utf-8"
    ) as file:
        json.dump(presentation, file, indent=2)

    logger.debug("Script done")

    return presentation


def create_image(index: int, slide: Dict, output: str = os.path.curdir):
    """Create the image for the slide

    The image will be saved in the output directory as `slide_*.png`. The image
    will be created by using the image prompt from the slide.

    Parameters
    ----------
    index : int
        The slide index
    slide : Dict
        The slide to create the image for
    output : str
        The output directory to use for the files, defaults to os.path.curdir

    Raises
    ------
    IndexError
        If the response is empty
    OSError
        If something goes wrong with the image download
    Exception
        If anything else goes wrong
    """
    logger.debug("Creating Slide %d: Image '%s'...", index, slide["image"])

    response = openai.Image.create(
        prompt=slide["image"],
        n=1,
        size="1024x1024",
        api_key=OPENAI_API_KEY,
    )
    image_url = response["data"][0]["url"]

    path = os.path.join(output, f"slide_{index:02d}.png")
    urllib.request.urlretrieve(image_url, path)

    logger.debug("Slide %d: Image done", index)


def create_audio(index: int, slide: Dict, output: str = os.path.curdir):
    """Create the audio for the slide

    The audio will be saved in the output directory as `slide_*.wav`. The audio
    will be created by using the text prompt from the slide.

    Parameters
    ----------
    index : int
        The slide index
    slide : Dict
        The slide to create the audio for
    output : str
        The output directory to use for the files, defaults to os.path.curdir

    Raises
    ------
    OSError
        If something goes wrong with the audio download
    Exception
        If anything else goes wrong
    """
    logger.debug(
        "Creating Slide %d: TTS (%s) '%s'...",
        index,
        slide["speaker"],
        slide["text"],
    )

    fk_you = fakeyou.FakeYou()

    try:
        fk_you.login(username=FAKEYOU_USERNAME, password=FAKEYOU_PASSWORD)
    except fakeyou.exception.InvalidCredentials:
        logger.warning("Invalid login credentials for FakeYou")
    except fakeyou.exception.TooManyRequests:
        logger.error("Too many requests for FakeYou")

    path = os.path.join(output, f"slide_{index:02d}.wav")
    fk_you.say(slide["text"], slide["speaker"]).save(path)

    logger.debug("Slide %d: TTS done", index)


def slide_gen(
    prompt: str,
    output: str = os.path.curdir,
    progress: Optional[Progress] = None,
):
    """Create the presentation, the slides, the audio and render the video

    The slides will be saved in the output directory as `slide_*.png` and
    `slide_*.wav`. The slides will be created by using the system prompt (from
    config) and the user prompt. The audio will be saved in the output
    directory as `slide_*.wav`. The audio will be created by using the text
    prompt from the slide. The video will be saved in the output directory as
    `video.mp4`. The video will be created by concatenating the images and
    audio files together. The subtitles will be saved in the output directory
    as `video.vtt`. The timing for each slide will be based on the `.wav`
    length.

    Parameters
    ----------
    prompt : str
        The user prompt to use
    output : str, optional
        The output directory to use for the files, by default os.path.curdir
    progress : Optional[Progress], optional
        The progress bar to use, by default None

    Raises
    ------
    Exception
        If something goes wrong
    """
    if progress is not None:
        presentation_job = progress.add_task(
            "[yellow]Generating script", total=None
        )

    try:
        presentation = create_script(prompt, output)
    except Exception:
        if progress is not None:
            progress.remove_task(presentation_job)
        raise

    if progress is not None:
        progress.update(presentation_job, completed=1, total=1)
        slides_job = progress.add_task(
            "Creating slides", total=2 * len(presentation)
        )

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for index, slide in enumerate(presentation):
            futures.append(executor.submit(create_image, index, slide, output))

        for future in concurrent.futures.as_completed(futures):
            try:
                _ = future.result()
            except Exception:
                if progress is not None:
                    progress.remove_task(presentation_job)
                    progress.remove_task(slides_job)
                raise

            if progress is not None:
                progress.advance(slides_job)

    # TODO: Will have to make this concurrent too for speed
    # FakeYou is kind of trash and needs to use serial requests
    for index, slide in enumerate(presentation):
        try:
            create_audio(index, slide, output)
        except Exception:
            if progress is not None:
                progress.remove_task(presentation_job)
                progress.remove_task(slides_job)
            raise

        if progress is not None:
            progress.advance(slides_job)

    try:
        create_vtt(output)
    except Exception:
        logger.warning("VTT creation failed")

    create_video(output)

    if progress is not None:
        progress.remove_task(presentation_job)
        progress.remove_task(slides_job)


if __name__ == "__main__":
    from rich.live import Live
    from rich.logging import RichHandler
    from rich.panel import Panel
    from rich.progress import BarColumn, SpinnerColumn, TextColumn

    from slide_twitch.util import OUTPUT, get_output_run

    rich_handler = RichHandler(rich_tracebacks=True)
    logging.root.addHandler(rich_handler)
    logging.root.setLevel(logging.ERROR)

    logger = logging.getLogger("slide_twitch")
    logger.setLevel(logging.DEBUG)

    _output = os.path.join(OUTPUT, str(get_output_run(OUTPUT)))

    job_progress = Progress(
        "{task.description}",
        SpinnerColumn(),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    )

    panel = Panel.fit(job_progress, title="Slide Twitch")

    with Live(panel, refresh_per_second=10):
        slide_gen(
            "how to create a house in minecraft",
            output=_output,
            progress=job_progress,
        )
