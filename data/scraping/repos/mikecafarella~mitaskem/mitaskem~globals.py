import os
import pathlib
import langchain
import langchain.cache
import langchain.globals

CACHE_BASE = pathlib.Path(f'{os.environ["HOME"]}/.cache/mitaskem/')
CACHE_BASE.mkdir(parents=True, exist_ok=True)
_LLM_CACHE_PATH = CACHE_BASE/'langchain_llm_cache.sqlite'
langchain.globals.set_llm_cache(langchain.cache.SQLiteCache(database_path=_LLM_CACHE_PATH))