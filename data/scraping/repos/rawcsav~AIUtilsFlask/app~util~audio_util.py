import asyncio
import os
import re
from pathlib import Path
from typing import Optional, Tuple, List

import openai
from pydub import AudioSegment

from app.config import INITIAL_PROMPT, SUPPORTED_FORMATS, MAX_AUDIO_FILE_SIZE


class TranscriptionFailedException(Exception):
    pass


def save_to_txt(content: str, filename: str):
    with open(filename, "w") as txt_file:
        txt_file.write(content)


def convert_to_mp3(file_path: str) -> Optional[str]:
    dirname, filename = os.path.split(file_path)
    basename, extension = os.path.splitext(filename)
    extension = extension.lower()
    mp3_file_path = os.path.join(dirname, f"{basename}.mp3")

    if extension in SUPPORTED_FORMATS:
        audio = AudioSegment.from_file(file_path, extension.lstrip("."))
        audio.export(mp3_file_path, format="mp3")
        return mp3_file_path
    else:
        return None


def transcribe_with_retry(
    audio_file: str,
    prompt: str,
    max_retries: int = 3,
    use_timestamps: bool = True,
    language: str = "en",
) -> Optional[str]:
    transcript = None

    for retry_count in range(max_retries):
        try:
            if use_timestamps:
                response = openai.Audio.transcribe(
                    "whisper-1",
                    audio_file,
                    response_format="srt",
                    language=language,
                    prompt=prompt,
                )
                transcript = response
            else:
                response = openai.Audio.transcribe(
                    "whisper-1", audio_file, language=language, prompt=prompt
                )
                transcript = response["text"]
            return transcript
        except Exception as e:
            print(f"Error: {e}. Retrying...")

    return None


def translate_with_retry(
    audio_file: str, prompt: str, max_retries: int = 3, use_timestamps: bool = True
) -> Optional[str]:
    transcript = None

    for retry_count in range(max_retries):
        try:
            if use_timestamps:
                response = openai.Audio.translate(
                    "whisper-1", audio_file, response_format="srt", prompt=prompt
                )
                translation = response

            else:
                response = openai.Audio.translate(
                    "whisper-1", audio_file, prompt=prompt
                )
                translation = response["text"]
            return translation
        except Exception as e:
            print(f"Error: {e}. Retrying...")

    return None


def parse_transcript_text(transcript_srt: str) -> str:
    parsed_timestamps = parse_timestamps(transcript_srt)
    transcribed_text = " ".join(text for timestamp, text in parsed_timestamps)
    return transcribed_text.strip()


def process_audio_file(
    file_path: str,
    whisper_directory: str,
    openai_api_key: str,
    use_timestamps: bool = True,
    language: str = "en",
    translate: bool = False,
) -> Optional[str]:
    temp_files = []  # List to keep track of temporary files created during processing

    file_path = Path(file_path)
    file_size = file_path.stat().st_size
    intermediate_txt = os.path.join(whisper_directory, "backup_output.txt")

    openai.api_key = openai_api_key
    try:
        # If the file is a compatible format and is smaller than 25MB
        if (
            file_size <= MAX_AUDIO_FILE_SIZE
            and file_path.suffix.lower() in SUPPORTED_FORMATS
        ):
            temp_files.append(file_path)  # Add to the temp files list
            with open(file_path, "rb") as audio_file:
                if translate:
                    transcript = translate_with_retry(
                        audio_file, INITIAL_PROMPT, use_timestamps=use_timestamps
                    )
                else:
                    transcript = transcribe_with_retry(
                        audio_file,
                        INITIAL_PROMPT,
                        language=language,
                        use_timestamps=use_timestamps,
                    )
                return transcript

        # If the file is larger than 25MB and is a compatible format but not an MP3
        elif (
            file_size > MAX_AUDIO_FILE_SIZE
            and file_path.suffix.lower() in SUPPORTED_FORMATS
            and file_path.suffix.lower() != ".mp3"
        ):
            mp3_file_path = convert_to_mp3(file_path)
            temp_files.append(mp3_file_path)  # Add to the temp files list
            file_path.unlink()

        # If the file is an MP3 and is larger than 25MB
        elif file_size > MAX_AUDIO_FILE_SIZE and file_path.suffix.lower() == ".mp3":
            mp3_file_path = file_path
            temp_files.append(mp3_file_path)  # Add to the temp files list
        else:
            return None

        # Check file size of mp3_file_path and split if necessary
        file_size = mp3_file_path.stat().st_size
        if file_size > MAX_AUDIO_FILE_SIZE:
            file_parts = split_large_file(mp3_file_path)
            mp3_file_path.unlink()

            transcripts_parts = []
            prompt = ""

            for part_idx, part_file in enumerate(file_parts):
                with open(part_file, "rb") as audio_file:
                    if translate:
                        transcript_part = translate_with_retry(
                            audio_file, prompt=prompt, use_timestamps=use_timestamps
                        )
                    else:
                        transcript_part = transcribe_with_retry(
                            audio_file,
                            prompt=prompt,
                            language=language,
                            use_timestamps=use_timestamps,
                        )
                    transcripts_parts.append(transcript_part)

                # Reset the prompt to an empty string for all parts except the first one
                prompt = INITIAL_PROMPT if part_idx == 0 else ""

                # Update prompt with transcribed text for the next part
                if transcript_part is not None:
                    if use_timestamps:
                        prompt += parse_transcript_text(transcript_part)
                    else:
                        prompt += transcript_part.strip()

                # Delete the part after transcribing
                Path(part_file).unlink()

            transcripts = ""
            timestamp_offset = 0

            if use_timestamps:
                for part_transcripts in transcripts_parts:
                    parsed_timestamps = parse_timestamps(part_transcripts)
                    last_end_timestamp = 0

                    for timestamp, text in parsed_timestamps:
                        match = re.search(
                            r"(\d+:\d+:\d+,\d+)\s+-->\s+(\d+:\d+:\d+,\d+)", timestamp
                        )
                        if match:
                            start_timestamp = match.group(1)
                            end_timestamp = match.group(2)
                            start_time = timestamp_to_ms(start_timestamp)
                            end_time = timestamp_to_ms(end_timestamp)
                        else:
                            print(f"Failed to find timestamp in timestamp: {timestamp}")
                            continue

                        new_start_time = ms_to_timestamp(start_time + timestamp_offset)
                        new_end_time = ms_to_timestamp(end_time + timestamp_offset)

                        new_timestamp = timestamp.replace(
                            start_timestamp, new_start_time
                        )
                        new_timestamp = new_timestamp.replace(
                            end_timestamp, new_end_time
                        )

                        transcripts += new_timestamp + "\n" + text + "\n\n"

                        last_end_timestamp = end_time

                    timestamp_offset += last_end_timestamp
                    save_to_txt(transcripts, intermediate_txt)
            else:
                transcripts = " ".join(transcripts_parts)

        return transcripts

    finally:
        # Cleanup temporary files
        for temp_file in temp_files:
            try:
                Path(temp_file).unlink()
            except Exception as e:
                print(f"Error deleting temp file {temp_file}: {e}")


def parse_timestamps(content: str) -> List[Tuple[str, str]]:
    parsed_timestamps = re.findall(
        r"(\d+\n\d+:\d+:\d+,\d+ --> \d+:\d+:\d+,\d+\n(.*?)(?:\n|\Z))",
        content,
        re.DOTALL,
    )
    if not parsed_timestamps:
        print(f"Error: Unable to parse timestamps from content: {content}")
    return parsed_timestamps


def is_full_sentence(text):
    return text.strip() and text[-1] in ".!?"


def format_timestamp(timestamp):
    timestamp = timestamp[:-4]
    h, m, s = map(int, timestamp.split(":"))
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    else:
        return f"{m:02d}:{s:02d}"


def process_timestamps(timestamps):
    new_timestamps = []
    timestamp_buffer = ""
    start_time = ""
    end_time = ""

    for timestamp, text in timestamps:
        if not start_time:
            start_time = re.search(r"(\d+:\d+:\d+,\d+) -->", timestamp).group(1)

        timestamp_buffer += " " + text.strip()

        if is_full_sentence(timestamp_buffer):
            end_time = re.search(r"--> (\d+:\d+:\d+,\d+)", timestamp).group(1)
            new_timestamps.append((start_time, end_time, timestamp_buffer.strip()))
            timestamp_buffer = ""
            start_time = ""
            end_time = ""

    return new_timestamps


def export_timestamps(timestamps: List[Tuple[str, str, str]], filename: str) -> None:
    with open(filename, "w") as f:
        for start, end, text in timestamps:
            start = format_timestamp(start)
            end = format_timestamp(end)
            f.write(f"[{start} - {end}] {text}\n\n")


def timestamp_to_ms(timestamp: str) -> int:
    hours, minutes, remaining = timestamp.split(":")
    seconds, milliseconds = remaining.split(",")
    total_ms = int(
        (int(hours) * 3600 + int(minutes) * 60 + float(seconds)) * 1000
        + int(milliseconds)
    )
    return total_ms


def ms_to_timestamp(ms: int) -> str:
    seconds, milliseconds = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{int(milliseconds):03d}"


def split_large_file(file_path: Path) -> List[str]:
    dirname, filename = os.path.split(str(file_path))
    split_duration_ms = 20 * 60 * 1000  # 24 minutes
    audio = AudioSegment.from_file(file_path)
    file_basename = file_path.stem
    file_parts = []

    for i in range(0, len(audio), split_duration_ms):
        part = audio[i : i + split_duration_ms]
        part_filename = os.path.join(
            dirname, f"{file_basename}_part{i // split_duration_ms}.mp3"
        )
        part.export(part_filename, format="mp3")
        file_parts.append(part_filename)

    return file_parts


def process_transcripts(content, output_file):
    parsed_timestamps = parse_timestamps(content)
    formatted_timestamps = process_timestamps(parsed_timestamps)
    export_timestamps(formatted_timestamps, output_file)


def transcribe_file(
    file_path: str,
    whisper_directory: str,
    openai_api_key: str,
    use_timestamps: bool = True,
    language: str = "en",
    translate: bool = False,
) -> str:
    # Validate the file format
    supported_formats = (".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm")
    if not file_path.endswith(supported_formats):
        raise ValueError("Unsupported audio file format.")

    openai.api_key = openai_api_key

    async def transcribe_async():
        loop = asyncio.get_running_loop()

        # Only processing one file, so we only have one task
        task = loop.run_in_executor(
            None,
            process_audio_file,
            file_path,
            whisper_directory,
            openai_api_key,
            use_timestamps,
            language,
            translate,
        )

        try:
            transcript = await asyncio.gather(task)
        except Exception as e:
            raise TranscriptionFailedException(str(e))

        basename = os.path.splitext(Path(file_path).name)[0]
        txt_filename = f"{basename}.txt"
        txt_file_path = os.path.join(whisper_directory, txt_filename)

        with open(txt_file_path, "w") as txt_file:
            txt_file.write(transcript[0])

        if use_timestamps:
            process_transcripts(transcript[0], txt_file_path)

        return txt_filename  # Add this line

    return asyncio.run(transcribe_async())
