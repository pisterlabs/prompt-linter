import langchain
from langchain.llms import OpenAI, Ollama
import dotenv

from pretty_print_callback_handler import PrettyPrintCallbackHandler

dotenv.load_dotenv()

from langchain import PromptTemplate

prompt = "What is DevSecOps ? Please be verbose and detailed in your answer."

llm = Ollama(model="llama2-uncensored")
pretty_callback = PrettyPrintCallbackHandler()
llm.callbacks = [pretty_callback]

from gptcache import Cache
from gptcache.manager.factory import manager_factory
from gptcache.processor.pre import get_prompt
from gptcache.adapter.api import init_similar_cache

from langchain.cache import GPTCache
import hashlib


def get_hashed_name(name):
    return hashlib.sha256(name.encode()).hexdigest()


def init_gptcache_old(cache_obj: Cache, llm: str):
    hashed_llm = get_hashed_name(llm)
    cache_obj.init(
        pre_embedding_func=get_prompt,
        data_manager=manager_factory(manager="map", data_dir=f"map_cache_{hashed_llm}"),
    )


def init_gptcache(cache_obj: Cache, llm: str):
    hashed_llm = get_hashed_name(llm)
    init_similar_cache(cache_obj=cache_obj, data_dir=f"similar_cache_{hashed_llm}")


langchain.llm_cache = GPTCache(init_gptcache)


result = llm(prompt)

print(result)
