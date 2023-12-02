import openai
from pydub import AudioSegment
from tempfile import NamedTemporaryFile
from openai.openai_object import OpenAIObject
from .export import segments_to_vtt, segments_to_srt


def transcribe(model, audio_file, response_format="json", audio_format="wav", **kwargs):
    """
    Transcribe the audio content to text.
    All arguments of openai.Audio.transcribe are supported

    :param file: The audio file object (not file name) to transcribe, in one of these formats: mp3, mp4, mpeg, mpga, m4a, wav, or webm.
    :param model: ID of the model to use.
    :param prompt: An optional text to guide the model's style or continue a previous audio segment. The prompt should match the audio language.
    :param response_format: The format of the transcript output, in one of these options: json, text, srt, verbose_json, or vtt. (default json)
    :param language: The language of the input audio. Supplying the input language in ISO-639-1 format will improve accuracy and latency.
    :param audio_format: The audio file format (mp3, wav, ...) default to wav, othert format require ffmpeg for other formats than wav
    """
    audio = AudioSegment.from_file(audio_file, format=audio_format)

    chunk_duration = 10 * 60

    result = None
    timing = 0
    for slice in _slice(model, audio, audio_format, chunk_duration, **kwargs):
        if slice is None:
            break
        if result is None:
            result = slice
        else:
            result = _merge(result, slice, timing)
        timing += chunk_duration
    return _export(result, response_format)


def _export(transcript, response_format):
    if transcript is None:
        return None  # TODO What to return ?
    elif response_format is None or response_format == "json":
        return transcript  # TODO aggregate JSON
    elif response_format == "verbose_json":
        return transcript
    elif response_format == "text":
        return transcript.text
    elif response_format == "vtt":
        return segments_to_vtt(transcript.segments)
    elif response_format == "srt":
        return segments_to_srt(transcript.segments)
    else:
        raise NotImplementedError(f"{response_format} is not a supported format")


def _slice(model, audio, audio_format, chunk_duration, **kwargs):
    for slice in audio[:: chunk_duration * 1000]:
        with NamedTemporaryFile(suffix="." + audio_format) as export:
            slice.export(export, format=audio_format)
            export.seek(0)
            yield openai.Audio.transcribe(
                model, export, response_format="verbose_json", **kwargs
            )


def _merge(o1, o2, timing):
    result = OpenAIObject(
        response_ms=o1.response_ms + o2.response_ms,
        task="transcribe",
    )
    result.text = o1.text + " " + o2.text
    if hasattr(o1, "segments"):
        segments = list(o1.segments)
        if len(segments) == 0:
            id = 0
        else:
            id = segments[-1]["id"] + 1
        for segment in o2.segments:
            segment["id"] = id
            segment["start"] += timing
            segment["end"] += timing
            id += 1
            segments.append(segment)
        result.segments = segments
    return result
