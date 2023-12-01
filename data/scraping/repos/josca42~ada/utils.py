import openai
from loguru import logger
from . import db_cache
from ada.config import config
from time import sleep


# openai.api_key = config["OPENAI_API_KEY"]

logger.add(config["DATA_DIR"] / "logs" / "ada.log")
logger = logger.opt(ansi=True)
logger.level("plan", no=33, color="<green>")
logger.level("data", no=33, color="<blue>")
logger.level("plot", no=33, color="<magenta>")


def openai_completion(
    prompt,
    stop=None,
    model="text-davinci-003",
    temperature=0,
    max_tokens=2_000,
    api_key=None,
    log_completion="",
):
    completion = db_cache.Completion(
        prompt=prompt,
        stop=stop,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
    )
    completion_stored = db_cache.crud_completion.get(completion.hash_id)
    if completion_stored:
        completion = completion_stored.copy()
    else:
        sleep(1)  # Sleep for one second to avoid rate limiting
        response = openai.Completion.create(
            prompt=prompt,
            model=model,
            temperature=temperature,
            stop=stop,
            max_tokens=max_tokens,
            api_key=api_key,
        )
        text = response["choices"][0].text
        completion.completion = text
        db_cache.crud_completion.create(completion)

    if log_completion:
        logger.log(log_completion, completion.completion)
    return completion.completion
