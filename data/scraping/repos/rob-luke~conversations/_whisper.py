from pathlib import Path
from typing import Dict

from openai import OpenAI
import whisper

client = OpenAI()


def process(
    audio_file: Path,
    model_name: str = "base.en",
    prompt: str | None = None,
    language: str = "en",
) -> Dict[str, str]:
    """Transcribe audio using Whisper.

    Parameters
    ----------
    audio_file : Path
        Path to the audio file.
    model_name : str, optional
        Name of the whisper model to use. To use the cloud service
        provided by OpenAI, use "openai.en", by default "base.en".
    prompt : str, optional
        Prompt to use for the transcription, by default None.
    language : str, optional
        Language to use for the transcription, by default "en".

    Returns
    -------
    transcript : Dict[str, str]
        Dictionary containing the audio transcript in whisper format.
    """
    if model_name == "openai.en":
        result = _cloud_whisper(audio_file, prompt=prompt, language=language)
    else:
        result = _local_whisper(
            audio_file, model_name=model_name, prompt=prompt, language=language
        )
    return result


def _cloud_whisper(
    audio_file: Path, prompt: str | None = None, language: str = "en"
) -> Dict[str, str]:
    """Transcribe audio using OpenAI's Whisper API.

    Parameters
    ----------
    audio_file : Path
        Path to the audio file.
    prompt : str, optional
        Prompt to use for the transcription, by default None.
    language : str, optional
        Language to use for the transcription, by default "en".

    Returns
    -------
    transcript : Dict[str, str]
        Dictionary containing the audio transcript in whisper format.
    """
    if prompt is None:
        result = client.audio.transcriptions.create(
            file=audio_file, model="whisper-1", response_format="verbose_json"
        )
    else:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
            prompt=prompt,
            language=language,
        )

    return result.model_dump()


def _local_whisper(
    audio_file: Path,
    model_name: str = "base.en",
    device: str = "cpu",
    prompt: str | None = None,
    language: str = "en",
) -> Dict[str, str]:
    """Transcribe audio using a local Whisper model.

    Parameters
    ----------
    audio_file : Path
        Path to the audio file.
    model_name : str, optional
        Name of the whisper model to use, available models are "base.en"
        and "large.en", by default "base.en".
    device : str, optional
        The device on which to run the model, either "cpu" or "gpu", by default "cpu".
    prompt : str, optional
        Prompt to use for the transcription, by default None.
    language : str, optional
        Language to use for the transcription, by default "en".

    Returns
    -------
    transcript : Dict[str, str]
        Dictionary containing the audio transcript in whisper format.
    """
    model = whisper.load_model(model_name, device=device)
    audio = whisper.load_audio(str(audio_file))
    result = model.transcribe(audio, prompt=prompt, language=language)
    return result
