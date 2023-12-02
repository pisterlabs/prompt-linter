import asyncio
import traceback
from functools import partial
from typing import Tuple

import openai

from src.llm.llm_base import LLMBase
from src.utils.logging import LOGGER

_LOGGER = LOGGER.bind(name="openai_gpt")


class Openai(LLMBase):
    def prepare(self):
        self.openai = openai
        self.openai.api_base = self.config.LLMs.openai.api_base
        self.openai.api_key = self.config.LLMs.openai.api_key

    def _sync_completion(self, prompt, **kwargs) -> Tuple[str, int] | None:
        """调用openai的Completion API
        :param prompt: 输入的文本（请确保格式化为openai的prompt格式）
        :param kwargs: 其他参数
        :return: 返回生成的文本和token总数 或 None
        """
        try:
            model = self.config.LLMs.openai.model
            resp = self.openai.ChatCompletion.create(
                model=model, messages=prompt, **kwargs
            )
            _LOGGER.debug(f"调用openai的Completion API成功，API返回结果为：{resp}")
            _LOGGER.info(
                f"调用openai的Completion API成功，本次调用中，prompt+response的长度为{resp['usage']['total_tokens']}"
            )
            return (
                resp["choices"][0]["message"]["content"],
                resp["usage"]["total_tokens"],
            )
        except Exception as e:
            _LOGGER.error(f"调用openai的Completion API失败：{e}")
            traceback.print_tb(e.__traceback__)
            return None

    async def completion(self, prompt, **kwargs) -> Tuple[str, int] | None:
        """调用openai的Completion API
        :param prompt: 输入的文本（请确保格式化为openai的prompt格式）
        :param kwargs: 其他参数
        :return: 返回生成的文本和token总数 或 None
        """
        loop = asyncio.get_event_loop()
        bound_func = partial(self._sync_completion, prompt, **kwargs)
        res = await loop.run_in_executor(None, bound_func)
        return res
