import os
from typing import Union, List, Dict

import graphsignal
import openai
from dotenv import load_dotenv
from langchain import LLMChain
from langchain.callbacks import PromptLayerCallbackHandler
from langchain.chat_models import ChatOpenAI

from src.core.Tools import load_llm, parse_roleMessages_to_prompts, extract_variable
from src.entity.QueryForm import RoleMessage

load_dotenv()


class LangchainLLM(object):
    def __init__(self):
        # Set this to your Zep server URL
        self.zep_api_url = os.environ.get('ZEP_API_URL')
        self.zep_api_key = os.environ.get('ZEP_API_KEY')

        # 配置llm模型
        self._temperature = float(os.environ.get('OPENAI_TEMPERATURE'))
        self._model_name = os.environ.get('OPENAI_MODEL_NAME')
        self._openai_api_key = os.environ.get('OPENAI_API_KEY')
        self._openai_proxy = os.environ.get('OPENAI_PROXY')
        self._openai_api_base = os.environ.get('OPENAI_API_BASE')
        self._max_tokens = int(os.environ.get('OPENAI_MAX_TOKENS'))
        self.llm = load_llm(
            self._temperature,
            self._model_name,
            self._openai_api_key,
            self._openai_proxy,
            self._openai_api_base,
            self._max_tokens
        )
        self.graphsignal_name = os.environ.get('GRAPHSIGNAL_NAME')

        graphsignal.configure(api_key='16ddccf7551cb0e1b4a20435c495e999', deployment=self.graphsignal_name)

    def load_llm(
            self,
            temperature,
            model_name,
            openai_api_key,
            openai_proxy,
            openai_api_base,
            frequency_penalty,
            presence_penalty,
            max_tokens,
            callbacks=None
    ):
        if callbacks is None:
            callbacks = []
        openai.api_key = os.environ["OPENAI_API_KEY"]
        if max_tokens is None:
            max_tokens = self._max_tokens
        if temperature is None:
            temperature = self._temperature
        if model_name is None:
            model_name = self._model_name
        if frequency_penalty is None:
            frequency_penalty = 0.0
        if presence_penalty is None:
            presence_penalty = 0.0

        return ChatOpenAI(
            temperature=float(temperature),
            model_name=model_name,
            streaming=True,
            openai_api_key=openai_api_key,
            openai_proxy=openai_proxy,
            openai_api_base=openai_api_base,
            max_tokens=int(max_tokens),
            callbacks=callbacks,
            model_kwargs={
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
            }
        )

    def get_chain(
            self,
            temperature: float = None,
            model_name: str = None,
            roleMessages: Union[List[RoleMessage], None] = None,
            question: Dict[str, str] = None,
            questionTemplate: str = None,
            frequency_penalty: float | None = None,
            presence_penalty: float | None = None,
            max_tokens: int | None = None,
            pl_tags=None,
    ):
        callbacks = []
        if pl_tags is not None:
            promptLayer = PromptLayerCallbackHandler(pl_tags=pl_tags)
            callbacks.append(promptLayer)

        _llm = self.load_llm(
            temperature,
            model_name,
            self._openai_api_key,
            self._openai_proxy,
            self._openai_api_base,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
            callbacks=callbacks
        )

        embedding_all = False
        if "{{embedding}}" in questionTemplate:
            embedding_all = True

        embedding_variable = extract_variable(questionTemplate)

        if embedding_all and len(embedding_variable) > 0:
            # 又要检索全部，又要筛选单个的情况下，抛出错误。
            raise Exception("embedding_all and embedding_variable can't be True at the same time.")

        input_variables = list(question.keys())
        text_qa_template = parse_roleMessages_to_prompts(roleMessages, False, input_variables, questionTemplate)
        return LLMChain(llm=_llm, verbose=True, prompt=text_qa_template)


llms = LangchainLLM()
