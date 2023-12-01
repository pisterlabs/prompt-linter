import time
from dotenv import load_dotenv

import langchain
from langchain.llms import OpenAI
from langchain.cache import InMemoryCache
from langchain.callbacks import get_openai_callback

load_dotenv()
langchain.llm_cache = InMemoryCache()

llm = OpenAI(model_name="text-davinci-002")

with get_openai_callback() as cb:
    start = time.time()
    result = llm("Whats the last day of the week?")
    end = time.time()
    print(result)
    print("--- cb")
    print(str(cb) + f" ({end - start:.2f} seconds)")

with get_openai_callback() as cb2:
    start = time.time()
    result2 = llm("Whats the last day of the week?")
    result3 = llm("Whats the last day of the week?")
    end = time.time()
    print(result2)
    print(result3)
    print("--- cb")
    print(str(cb2) + f" ({end - start:.2f} seconds)")
