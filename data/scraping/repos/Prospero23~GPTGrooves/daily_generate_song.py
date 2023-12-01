import datetime

import langchain
from langchain.cache import SQLiteCache
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from music_generator.db import insert_song
from music_generator.generate_markup import generate_markup
from music_generator.generate_song import generate_song
from music_generator.music_generator_types.base_song_types import Config, SongRecord
from music_generator.utilities.logs import get_logger
from music_generator.utilities.set_langchain_environment import (
    set_langchain_environment,
)

logger = get_logger(__name__)


def daily_generate_song_and_persist(config: Config) -> None:
    """
    Generate a bar using each of the LLMs and save them to the database.
    """

    musical_markup = generate_markup(
        song_description="""Create an outline for a house music track""".strip(),
        llm=ChatOpenAI(
            openai_api_key=config.openai_api_key,
            model="gpt-4",
            temperature=0.70,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        ),
    )

    song = generate_song(
        llm=ChatOpenAI(
            openai_api_key=config.openai_api_key,
            model="gpt-4",
            temperature=0.0,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        ),
        musical_markup=musical_markup,
    )

    song_record = SongRecord(
        song=song, created_at_utc=datetime.datetime.utcnow().isoformat()
    )

    insert_song(
        config=config,
        song_record=song_record,
    )


if __name__ == "__main__":
    from dotenv import dotenv_values

    config = Config(**dotenv_values())  # type: ignore
    set_langchain_environment(config=config)
    langchain.llm_cache = (
        SQLiteCache(database_path=config.llm_cache_filename)
        if config.llm_cache_filename
        else None
    )

    daily_generate_song_and_persist(config=config)
