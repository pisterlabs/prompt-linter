#!/usr/bin/env python3
"""
Create cheatsheets from videos using the power of OpenAI
"""

import sys
import os
import argparse
import mimetypes
import tempfile
import datetime
import json

from pathlib import Path
from pydub import AudioSegment
from yt_dlp import YoutubeDL

import pysrt
import openai


TWENTYFIVE_MB = 26214400
TEMP_DIR = Path(tempfile.gettempdir())
DEFAULT_SUMMARY_PROMPT = (
    "Create a cheatsheet out of the following transcript in less than 50 words: \n"
)
CONTINUE_SUMMARY_PROMPT = (
    "Continue with the next part of the same transcript,"
    + "use the same style as before: \n"
)
SYSTEM_PROMPT = {
    "role": "system",
    "content": "You are a helpful assistant who summarizes with bullet points.",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("media", help="Path or URL to media file", type=str)
    parser.add_argument(
        "--api-key",
        help="OpenAI API key (default: read from OPENAI_API_KEY environment variable)",
        default=os.environ.get("OPENAI_API_KEY"),
    )
    parser.add_argument(
        "--transcription-prompt",
        help="Prompt to use for transcribing the video (e.g. What the video is about)",
        default="",
    )
    parser.add_argument(
        "--summary-prompt",
        help="Override the default prompt for summarizing the video",
        default=DEFAULT_SUMMARY_PROMPT,
    )
    parser.add_argument(
        "--frequency",
        help="How often (in sec) to create summaries of the video (default: 60)",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--model",
        help="OpenAI model to use (default: gpt-3.5-turbo)",
        default="gpt-3.5-turbo",
    )
    parser.add_argument(
        "--language",
        help="Language of the input media for transcribing"
        + " (default: en, must be in ISO 639-1 format and supported by OpenAI's Whisper API)."
        + " For translating, the language is automatically detected"
        + " and the output language is always English.",
    )
    parser.add_argument(
        "--output",
        help="Path to the output file (default: only stdout)",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-format",
        help="Output format, choose between: text (default), json",
        default="text",
    )
    parser.add_argument(
        "--run-whisper-locally",
        help="Run the Whisper API locally (default: False)",
        required=False,
        action="store_true",
    )
    args = parser.parse_args()

    if Path(args.media).is_file():
        args.media = Path(args.media)
    else:
        audio_codec = "m4a"
        audio_fname = Path("skonaki_audio_from_youtube." + audio_codec)
        extracted_audio = TEMP_DIR / audio_fname
        ydl_opts = {
            "outtmpl": str(extracted_audio.with_suffix("")),
            "overwrites": True,
            "format": "m4a/bestaudio/best",
            "postprocessors": [
                {  # Extract audio using ffmpeg
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": audio_codec,
                }
            ],
        }
        with YoutubeDL(ydl_opts) as ydl:
            ydl_code = ydl.download([args.media])
            if ydl_code != 0:
                print(
                    "Unable to download media file from: "
                    + args.media
                    + " error: "
                    + str(ydl_code)
                )
            print("Downloaded from: " + args.media + " to: " + str(extracted_audio))
            args.media = extracted_audio

    exit_code, exit_message = generate_summary(
        media=args.media,
        api_key=args.api_key,
        transcription_prompt=args.transcription_prompt,
        summary_prompt=args.summary_prompt,
        model=args.model,
        language=args.language,
        frequency=args.frequency,
        output_path=args.output,
        output_format=args.output_format,
        use_local_whisper=args.run_whisper_locally,
    )
    print(exit_message)
    return exit_code


def generate_summary(
    media: Path,
    api_key: str = os.environ.get("OPENAI_API_KEY"),
    transcription_prompt: str = "",
    summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
    model: str = "gpt-3.5-turbo",
    language: str = "en",
    frequency: int = 60,
    output_path: Path = None,
    output_format: str = "text",
    use_local_whisper: bool = False,
):
    if not media.is_file():
        exit_message = f"Media file {media} does not exist"
        return (1, exit_message)

    if not api_key:
        exit_message = (
            "OpenAI API key is required, none provided or found in environment"
        )
        return (1, exit_message)

    audio = get_audio(media)
    audio_size = audio.stat().st_size
    if audio_size > TWENTYFIVE_MB:
        print(
            f"Audio file is too large: {audio_size / 1000000}MB"
            + ". It must be less than 25MB, attempting to downsample"
        )
        audio = downsample_audio(audio, TWENTYFIVE_MB)
        audio_size = audio.stat().st_size
    print(f"Audio file size in MB: {audio_size / 1000000}")

    if use_local_whisper:
        try:
            import whisper  # pylint: disable=import-outside-toplevel
        except ImportError:
            error_message = (
                "Error: Failed to import whisper. "
                + "Please install the correct dependencies from requirements-local-whisper.txt"
            )
            return (1, error_message)

        print("Transcribing using Whisper locally")
        local_whisper_model = whisper.load_model("base")
        loaded_audio = whisper.load_audio(audio)
        result = whisper.transcribe(
            model=local_whisper_model,
            audio=loaded_audio,
            language=language,
            prompt=transcription_prompt,
        )
        # Need to use the get_writer() to get the output into srt format
        # https://github.com/openai/whisper/discussions/758
        writer = whisper.utils.get_writer("srt", ".")
        # "None" set for options following the
        # answer here: https://github.com/openai/whisper/discussions/1229#discussioncomment-7091769
        writer(
            result,
            audio,
            {"max_line_width": None, "max_line_count": None, "highlight_words": False},
        )
        # The writer() saves the file as audio.srt, and so the following
        # lines are used to read the file into a string.
        with open("audio.srt", "r") as f:
            transcript = f.read()
    else:
        openai.api_key = api_key
        print("Transcribing using OpenAI's Whisper")
        with open(audio, "rb") as f:
            transcript = openai.Audio.transcribe(
                "whisper-1",
                f,
                response_format="srt",
                language=language,
                prompt=transcription_prompt,
            )

    subs = pysrt.from_string(transcript)
    # Break the transcript into chunks based on the frequency
    chunks = []
    chunk = []
    chunk_beginning = subs[0].start.ordinal
    for sub in subs:
        chunk.append(sub)
        if sub.start.ordinal - chunk_beginning > frequency * 1000:
            chunks.append((chunk, chunk_beginning))
            chunk = []
            chunk_beginning = sub.start.ordinal
    if chunk:
        chunks.append((chunk, chunk_beginning))

    messages = [SYSTEM_PROMPT]
    cheatsheet = {}
    current_chunk = 1
    for subtitle_chunk, chunk_timestamp in chunks:
        # Convert the chunk to text
        text = "\n".join([sub.text for sub in subtitle_chunk])

        # Count the number of characters in messages
        characters_per_token = 4
        max_tokens = get_max_tokens(model)
        if get_characters(messages) > max_tokens * characters_per_token:
            # Keep only the first message (system prompt) and the last message (assistant response)
            print("Reached the max number of tokens, resetting messages")
            assert len(messages) > 2
            messages = [messages[0], messages[-1]]
            # There's a chance that the assistant response is too long, so trim
            if get_characters(messages) > max_tokens * characters_per_token:
                print("The last message is too long, trim it to the max length")
                messages[-1]["content"] = messages[-1]["content"][
                    max_tokens * characters_per_token :
                ]
                messages[-1]["content"] = "..." + messages[-1]["content"]

        continue_or_first_prompt = (
            CONTINUE_SUMMARY_PROMPT if len(messages) > 1 else summary_prompt
        )
        summary_prompt = continue_or_first_prompt + "\n" + text
        messages.append(
            {
                "role": "user",
                "content": text,
            },
        )

        print(
            f"Summarizing using OpenAI's {model} model. Part {current_chunk} of {len(chunks)}."
        )
        current_chunk += 1
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.6,
        )
        gpt_response = response.choices[0].message.content
        # Format timestamp in hh:mm:ss format
        chunk_timedelta = datetime.timedelta(milliseconds=chunk_timestamp)
        chunk_timedelta_str = str(chunk_timedelta).split(".", maxsplit=1)[0]
        # If hours is only 1 digit, add a leading 0
        if len(chunk_timedelta_str.split(":")[0]) == 1:
            chunk_timedelta_str = "0" + chunk_timedelta_str

        cheatsheet[chunk_timedelta_str] = gpt_response
        messages.append(
            {
                "role": "assistant",
                "content": gpt_response,
            },
        )

    formatted_output = format_output(cheatsheet, output_format)

    if output_path:
        output_path.write_text(formatted_output)
        print(f"Saved cheatsheet to {output_path.resolve()}")

    exit_message = "\n\n\n" + formatted_output
    return (0, exit_message)


def format_output(cheatsheet: dict, output_format: str):
    if output_format == "json":
        return json.dumps(cheatsheet, indent=4)
    # Return as <timestamp>\n<summary> for each timestamp and summary
    return "\n".join(
        [f"{timestamp}\n{summary}" for timestamp, summary in cheatsheet.items()]
    )


def get_characters(messages: list):
    return sum(len(message["content"]) for message in messages)


def get_max_tokens(model: str):
    if model == "gpt-4":
        return 7000

    return 3000


def get_audio(media: Path):
    print(f"Getting audio from {media}")
    file_type = mimetypes.guess_type(media)[0]
    if file_type == "audio":
        print("Media is already audio, no need to convert")
        return media

    audio = TEMP_DIR / "audio.mp3"
    AudioSegment.from_file(media).set_channels(1).export(
        audio, format="mp3", bitrate="128k"
    )
    print(f"Split audio file and saved to {audio}")
    return audio


def downsample_audio(audio: Path, max_size: int = TWENTYFIVE_MB):
    print(f"Downsampling audio from {audio}")
    bitrates = ["64k", "32k", "16k"]
    for bitrate in bitrates:
        downsampled = TEMP_DIR / "audio_downsampled.mp3"
        AudioSegment.from_file(audio).set_channels(1).export(
            downsampled, format="mp3", bitrate=bitrate
        )
        if downsampled.stat().st_size < max_size:
            print(
                f"Downsampled audio file and saved to {downsampled} with bitrate {bitrate}"
            )
            return downsampled

    print("Unable to downsample audio file, it needs to be split into smaller chunks")
    print("Open a feature request on GitHub if you need this feature")
    raise RuntimeError("Unable to downsample audio file")


if __name__ == "__main__":
    sys.exit(main())
