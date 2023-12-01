# TODO(scott): clean up

import functools

import openai
import langchain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import utils.common
import config
from chat.base import ChatOutputParser
from chat.rephrase_widget_search2 import TEMPLATE

from gpt_index.utils import ErrorToRetry, retry_on_exceptions_with_backoff

utils.common.set_api_key()


from langchain.cache import SQLiteCache
# langchain.llm_cache = SQLiteCache(database_path=".langchain.db")


@functools.cache
def get_llm_chain():
    llm = OpenAI(
        temperature=0.0,
        max_tokens=-1,
    )
    output_parser = ChatOutputParser()
    prompt = PromptTemplate(
        input_variables=["task_info", "chat_history", "question"],
        template=TEMPLATE,
        output_parser=output_parser,
    )
    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    return chain


def get_llm_output(user_input, task_info, history_string):
    example = {
        "task_info": task_info,
        "chat_history": history_string,
        "question": user_input,
        "stop": ["Input", "User"],
    }
    llm_chain = get_llm_chain()
    try:
        output = retry_on_exceptions_with_backoff(
            lambda: llm_chain.run(example),
            [ErrorToRetry(openai.error.RateLimitError)],
        )
    except openai.error.InvalidRequestError:
        return None

    return output
