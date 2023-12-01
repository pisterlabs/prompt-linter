#!/usr/bin/env python3
"""
Voice file (ogg) to MP3 to Whisper to OpenAI to Text to Voice file (mp3)

Usage:
python3 voice_to_openai.py -i 5ee7cb98-4004-4521-a697-3fdb3e939f24.ogg

"""

import logging
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path

import gtts
import whisper
from py_executable_checklist.workflow import WorkflowBase, run_command, run_workflow

from common_utils import setup_logging
from openai_api import completions

model = whisper.load_model("medium")


# Workflow steps


class ConvertToMP3(WorkflowBase):
    """
    Convert OGG audio file to MP3.
    """

    ogg_file_path: Path

    def execute(self):
        mp3_file_path = self.ogg_file_path.with_suffix(".mp3")
        command = f"ffmpeg -hide_banner -loglevel error -y -i {self.ogg_file_path} {mp3_file_path}"
        run_command(command)
        return {"mp3_file_path": mp3_file_path}


class ConvertToText(WorkflowBase):
    """
    Convert MP3 file to text using Whisper API.
    """

    mp3_file_path: Path

    def execute(self):
        recognized_text = model.transcribe(audio=self.mp3_file_path.as_posix(), fp16=False)
        return {"recognized_text": recognized_text["text"]}


class SendToOpenAICompletionAPI(WorkflowBase):
    """
    Send recognized text to OpenAI Completion API and return response.
    """

    recognized_text: str

    def execute(self):
        response = completions(prompt=self.recognized_text)
        return {"completions_response": response}


class TextToMP3(WorkflowBase):
    """
    Convert text to MP3 using text-to-speech library.
    """

    completions_response: str
    mp3_file_path: Path

    def execute(self):
        generated_mp3_file_path = self.mp3_file_path.with_suffix(".generated.mp3")
        tts = gtts.gTTS(self.completions_response)
        tts.save(generated_mp3_file_path.as_posix())
        return {"output_mp3_file_path": generated_mp3_file_path}


# Workflow definition


def workflow():
    return [
        ConvertToMP3,
        ConvertToText,
        SendToOpenAICompletionAPI,
        TextToMP3,
    ]


# Boilerplate


def parse_args():
    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--ogg-file-path", type=Path, required=True, help="Input audio file in ogg format")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        dest="verbose",
        help="Increase verbosity of logging output. Display context variables between each step run",
    )
    return parser.parse_args()


def run(ogg_file_path):
    context = {
        "ogg_file_path": ogg_file_path,
    }
    run_workflow(context, workflow())
    return context["output_mp3_file_path"]


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.verbose)
    generated_mp3_file = run(args.ogg_file_path)
    logging.info(f"Generated MP3 file saved to {generated_mp3_file}")
