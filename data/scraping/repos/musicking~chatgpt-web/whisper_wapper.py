import openai
from backoff import on_exception, expo
from io import BytesIO
from typing import Optional
from fastapi import UploadFile
import concurrent.futures
import asyncio
import traceback
from loguru import logger
from errors import Errors


def upload_file_to_file_obj(upload_file: UploadFile, file_obj: Optional[BytesIO] = None):
    if file_obj is None:
        file_obj = BytesIO()
    file_obj.write(upload_file.file.read())
    file_obj.seek(0)
    file_obj.name = upload_file.filename
    return file_obj


async def process_audio(audio, model="whisper-1"):
    try:
        file = upload_file_to_file_obj(audio)

        params = dict(
            model=model,
            file=file
        )
        transcript = await _create_async(params)
        if transcript is None:
            yield Errors.SOMETHING_WRONG_IN_OPENAI_WHISPER_API.value
            return

        prompt = transcript["text"]
        logger.debug("audio prompt: {}".format(prompt))
        del audio
        del file
    except:
        err = traceback.format_exc()
        logger.error(err)
        yield Errors.SOMETHING_WRONG.value
        return

    yield prompt


@on_exception(expo, openai.error.RateLimitError, max_tries=5)
def _create(params):
    return openai.Audio.transcribe(**params)


async def _create_async(params):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                executor, _create, params
            )
        except:
            err = traceback.format_exc()
            logger.error(err)
            # 这里处理 openai.error.RateLimitError 之外的错误
            return None
    return result
