import asyncio
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Optional, cast

from superlesson.storage import LessonFile, Slide, Slides
from superlesson.storage.slide import TimeFrame
from superlesson.storage.store import Store
from superlesson.storage.utils import diff_words

from .step import Step, step

logger = logging.getLogger("superlesson")


@dataclass
class Segment:
    text: str
    start: float
    end: float


@dataclass
class Prompt:
    body: str
    slide: int = 0


class Transcribe:
    """Class to transcribe a lesson."""

    _bucket_name = "lesson-audios"

    def __init__(self, slides: Slides, transcription_source: LessonFile):
        from dotenv import load_dotenv

        load_dotenv()

        self._transcription_source = transcription_source
        self.slides = slides

    @step(Step.transcribe)
    def single_file(self, model_size: str, local: bool):
        bench_start = time.time()

        audio_path = self._transcription_source.extract_audio(overwrite=True)

        if not local and os.getenv("REPLICATE_API_TOKEN"):
            s3_url = self._upload_file_to_s3(audio_path)

            if not model_size.startswith("large"):
                logger.info("Ignoring model size and using large instead")

            segments = self._transcribe_with_replicate(s3_url)
        else:
            if not local and (
                input(
                    "Replicate token not set. Do you want to run Whisper locally? (y)es/(N)o"
                )
                != "y"
            ):
                msg = "Couldn't run transcription."
                raise Exception(msg)
            segments = self._local_transcription(audio_path, model_size)

        for segment in segments:
            self.slides.append(
                Slide(segment.text, TimeFrame(segment.start, segment.end))
            )

        bench_duration = time.time() - bench_start
        logger.info(f"Transcription took {bench_duration} to finish")

    @classmethod
    def _transcribe_with_replicate(cls, url: str) -> list[Segment]:
        import replicate

        logger.info("Running replicate")
        output = replicate.run(
            "isinyaaa/whisperx:f2f27406afdd5f2bd8aab728e9c50eec8378dcf67381b42009051a156d83ddba",
            input={
                "audio": url,
                "language": "pt",
                "batch_size": 13,
                "align_output": True,
            },
        )
        logger.info("Replicate finished")
        assert isinstance(output, dict), "Expected a dict"
        segments = []
        for segment in output["word_segments"]:
            if "start" in segment:
                segments.append(
                    Segment(segment["word"], segment["start"], segment["end"])
                )
            elif len(segments) != 0:
                segments[-1].text += " " + segment["word"]
        return segments

    @classmethod
    def _upload_file_to_s3(cls, path: Path) -> str:
        import boto3
        from botocore.exceptions import ClientError

        s3 = boto3.client("s3")

        # TODO: we should salt it to improve privacy
        # ideally, we should also encrypt the data, or figure out a way to
        # authenticate from replicate
        with open(path, "rb") as file:
            data = file.read()
            s3_name = sha256(data).hexdigest()

        s3_path = f"https://{cls._bucket_name}.s3.amazonaws.com/{s3_name}"

        try:
            s3.head_object(Bucket=cls._bucket_name, Key=s3_name)
            return s3_path
        except ClientError:
            pass

        logger.info(f"Uploading file {file} to S3")

        s3.upload_file(path, cls._bucket_name, s3_name)

        logger.info(f"{file} uploaded to S3 as {s3_name}")
        return s3_path

    @classmethod
    def _local_transcription(cls, transcription_path: Path, model_size: str):
        from faster_whisper import WhisperModel

        if cls._has_nvidia_gpu():
            model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
        else:
            threads = os.cpu_count() or 4
            model = WhisperModel(
                model_size, device="cpu", cpu_threads=threads, compute_type="auto"
            )

        segments, info = model.transcribe(
            str(transcription_path),
            beam_size=5,
            language="pt",
            vad_filter=True,
        )

        logger.info(
            f"Detected language {info.language} with probability {info.language_probability}"
        )

        return cls._run_with_pbar(segments, info)

    @staticmethod
    def _has_nvidia_gpu():
        try:
            subprocess.check_output("nvidia-smi")
            return True
        except Exception:
            return False

    # taken from https://github.com/guillaumekln/faster-whisper/issues/80#issuecomment-1565032268
    @classmethod
    def _run_with_pbar(cls, segments, info):
        import io
        from threading import Thread

        from tqdm import tqdm

        duration = round(info.duration)
        bar_f = "{percentage:3.0f}% |  {remaining}  | {rate_noinv_fmt}"
        print("  %  | remaining |  rate")

        capture = io.StringIO()  # capture progress bars from tqdm

        with tqdm(
            file=capture,
            total=duration,
            unit=" audio seconds",
            smoothing=0.00001,
            bar_format=bar_f,
        ) as pbar:
            global timestamp_prev, timestamp_last
            timestamp_prev = 0  # last timestamp in previous chunk
            timestamp_last = 0  # current timestamp
            last_burst = 0.0  # time of last iteration burst aka chunk
            set_delay = (
                0.1  # max time it takes to iterate chunk & minimum time between chunks
            )
            jobs = []
            transcription_segments = []
            for segment in segments:
                transcription_segments.append(segment)
                logger.info(
                    "[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text)
                )
                timestamp_last = round(segment.end)
                time_now = time.time()
                if time_now - last_burst > set_delay:  # catch new chunk
                    last_burst = time_now
                    job = Thread(
                        target=cls._pbar_delayed(set_delay, capture, pbar),
                        daemon=False,
                    )
                    jobs.append(job)
                    job.start()

            for job in jobs:
                job.join()

            if timestamp_last < duration:  # silence at the end of the audio
                pbar.update(duration - timestamp_last)
                print(
                    "\33]0;" + capture.getvalue().splitlines()[-1] + "\a",
                    end="",
                    flush=True,
                )
                print(capture.getvalue().splitlines()[-1])

        return transcription_segments

    @staticmethod
    def _pbar_delayed(set_delay, capture, pbar):
        """Gets last timestamp from chunk"""

        def pbar_update():
            global timestamp_prev
            time.sleep(set_delay)  # wait for whole chunk to be iterated
            pbar.update(timestamp_last - timestamp_prev)
            print(
                "\33]0;" + capture.getvalue().splitlines()[-1] + "\a",
                end="",
                flush=True,
            )
            print(capture.getvalue().splitlines()[-1])
            timestamp_prev = timestamp_last

        return pbar_update

    @step(Step.replace, Step.merge)
    def replace_words(self):
        replacements_path = self._transcription_source.path / "replacements.txt"
        if not replacements_path.exists():
            logger.warning(
                f"{replacements_path} doesn't exist, so no replacements will be done"
            )
            return

        pattern = re.compile(r'\s*([^"]*[^"\s])')

        lines = replacements_path.read_text().split("\n")
        for line in lines:
            if line.strip() == "":
                continue
            words = line.split("->")
            if len(words) != 2:
                logger.warning(f"Invalid line: {line}")
                continue
            word = pattern.search(words[0]).group(1)
            rep = pattern.search(words[1]).group(1)
            logger.debug("Replacing %s with %s", word, rep)
            for slide in self.slides:
                slide.transcription = re.sub(
                    r"\b%s\b" % re.escape(word),
                    rep,
                    slide.transcription,
                    flags=re.IGNORECASE,
                )

    @staticmethod
    def _diff_gpt(before: str, after: str):
        file1 = Store.temp_save(before)
        file2 = Store.temp_save(after)

        diff_words(file1, file2)

    @step(Step.improve, Step.merge)
    def improve_punctuation(self):
        context = """O texto a seguir precisa ser preparado para impressão.
        - formate o texto, sem fazer modificações de conteúdo.
        - corrija qualquer erro de digitação ou de grafia.
        - faça as quebras de paragrafação que forem necessárias.
        - coloque as pontuações adequadas.
        - a saída deve ser somente a resposta, sem frases como "aqui está o texto revisado e formatado".
        - NÃO FAÇA NENHUMA MODIFICAÇÃO DE CONTEÚDO, SOMENTE DE FORMATAÇÃO.
        """

        # margin = 20  # to avoid errors
        # GPT-3.5-turbo-1106 takes in at most 16k tokens, but it only outputs 4k tokens, so we use
        # it as the maximum and ignore margin and context size for now
        max_input_tokens = 2**12  # - self._count_tokens(context)) // 2 - margin
        logger.debug(f"Max input tokens: {max_input_tokens}")

        logger.debug("Splitting into prompts")
        prompts = self._split_into_prompts(
            [slide.transcription for slide in self.slides],
            max_input_tokens,
        )

        bench_start = time.time()

        completions = asyncio.run(self._complete_with_chatgpt(prompts, context))

        bench_duration = time.time() - bench_start
        logger.info(f"ChatGPT requests took {bench_duration} to finish")

        last_slide = 0
        improved_transcription = []
        for prompt, completion in zip(prompts, completions, strict=True):
            text = prompt.body
            similarity_ratio = self._calculate_difference(text, completion)
            # different = 0 < similarity_ratio < 1 = same
            if similarity_ratio < 0.40:
                logger.info("The text was not improved by ChatGPT-3.5-turbo.")
                logger.debug(f"Similarity: {similarity_ratio}")
                logger.debug(f"Diffing GPT output for slide {prompt.slide}")
                if logger.isEnabledFor(logging.DEBUG):
                    self._diff_gpt(text, completion)
                continue
            if prompt.slide != last_slide:
                self.slides[last_slide].transcription = " ".join(improved_transcription)
                improved_transcription = []
                last_slide = prompt.slide
            improved_transcription.append(completion)

    @staticmethod
    def _count_tokens(text: str) -> int:
        import tiktoken

        # Based on: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        tokens_per_message = 4
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text)) + tokens_per_message

    @classmethod
    def _split_into_prompts(
        cls, transcriptions: list[str], max_tokens: int
    ) -> list[Prompt]:
        prompts = []
        for i, transcription in enumerate(transcriptions):
            chunks = []
            for period in cls._split_in_periods(transcription):
                tokens = cls._count_tokens(period)
                if tokens > max_tokens:
                    logger.debug(f"Splitting text with {tokens} tokens")
                    words = period.split()
                    chunks.extend(cls._merge_chunks(words, max_tokens))
                else:
                    chunks.append(period)

            merged = cls._merge_chunks(chunks, max_tokens)
            for text in merged:
                prompts.append(Prompt(text, i))

        return prompts

    @classmethod
    def _split_in_periods(cls, text: str) -> list[str]:
        import re

        splits = re.split(r"([.?!])", text)
        if len(splits) == 1:
            return splits

        if splits[0] == "":
            splits = splits[1:]

        if splits[-1] == "":
            splits = splits[:-1]

        # merge punctuation with previous sentence
        # we iterate over len(splits) - 1 because, if it's even, we get the same result, and if
        # it's odd, we skip the invalid iteration at the end
        periods = []
        for i in range(0, len(splits) - 1, 2):
            periods.append(splits[i] + splits[i + 1])

        if len(splits) % 2 == 1:
            periods.append(splits[-1])

        return periods

    @classmethod
    def _merge_chunks(cls, chunks: list[str], max_tokens: int) -> list[str]:
        merged = []
        total_tokens = 0
        start = 0
        for i, chunk in enumerate(chunks):
            tokens = cls._count_tokens(chunk)
            if total_tokens + tokens > max_tokens:
                merged.append(" ".join(chunks[start:i]))

                start = i
                total_tokens = tokens
            else:
                total_tokens += tokens
        merged.append(" ".join(chunks[start:]))

        return merged

    @classmethod
    async def _complete_with_chatgpt(
        cls, prompts: list[Prompt], context: str
    ) -> list[str]:
        from openai import AsyncOpenAI, OpenAIError

        try:
            client = AsyncOpenAI()
        except OpenAIError as e:
            raise Exception(
                "Please review README.md for instructions on how to set up your OpenAI token"
            ) from e

        tasks = [
            asyncio.create_task(cls._complete_prompt(client, prompt.body, context))
            for prompt in prompts
        ]

        results = await asyncio.gather(*tasks)
        completions = []
        for result in results:
            if result is None:
                break
            cast(str, result)
            completions.append(result)
        logger.debug("ChatGPT completed prompt successfully")

        return completions

    @classmethod
    async def _complete_prompt(cls, client, prompt: str, context: str) -> Optional[str]:
        import openai

        messages = [
            {"role": "system", "content": context},
            {"role": "user", "content": prompt},
        ]
        logger.debug("Completing prompt: %s", prompt)
        try:
            completion = await client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                # model="gpt-4",
                messages=messages,
                n=1,
                temperature=0.1,
            )
            logger.debug("ChatGPT response: %s", completion.choices[0].message.content)
            return completion.choices[0].message.content
        except openai.RateLimitError:
            return None

    @staticmethod
    def _calculate_difference(paragraph1, paragraph2):
        import difflib

        words1 = paragraph1.split()
        words2 = paragraph2.split()
        similarity_ratio = difflib.SequenceMatcher(None, words1, words2).ratio()
        return similarity_ratio
