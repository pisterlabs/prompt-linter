import traceback
from typing import Tuple

import anthropic

from src.llm.llm_base import LLMBase
from src.llm.templates import Templates
from src.utils.logging import LOGGER
from src.utils.prompt_utils import parse_prompt

_LOGGER = LOGGER.bind(name="aiproxy_claude")


class AiproxyClaude(LLMBase):
    def prepare(self):
        mask_key = self.config.LLMs.aiproxy_claude.api_key[:-5] + "*****"
        _LOGGER.info(
            f"初始化AIProxyClaude，api_key为{mask_key}，api端点为{self.config.LLMs.aiproxy_claude.api_base}"
        )

    async def completion(self, prompt, **kwargs) -> Tuple[str, int] | None:
        """调用claude的Completion API
        :param prompt: 输入的文本（请确保格式化为openai的prompt格式）
        :param kwargs: 其他参数
        :return: 返回生成的文本和token总数 或 None
        """
        try:
            claude = anthropic.AsyncAnthropic(
                api_key=self.config.LLMs.aiproxy_claude.api_key,
                base_url=self.config.LLMs.aiproxy_claude.api_base,
            )
            resp = await claude.completions.create(
                prompt=prompt,
                max_tokens_to_sample=1000,
                model=self.config.LLMs.aiproxy_claude.model,
                **kwargs,
            )
            _LOGGER.debug(f"调用claude的Completion API成功，API返回结果为：{resp}")
            _LOGGER.info(
                f"调用claude的Completion API成功，本次调用中，prompt+response的长度为{resp.model_dump()['usage']['total_tokens']}"
            )
            if not resp.completion.startswith('{"'):
                # claude好像是直接从assistant所给内容之后续写的，给它加上缺失的前缀
                resp.completion = '{"' + resp.completion
            return (
                resp.completion,
                resp.model_dump()["usage"]["total_tokens"],
            )
        except Exception as e:
            _LOGGER.error(f"调用claude的Completion API失败：{e}")
            traceback.print_tb(e.__traceback__)
            return None

    @staticmethod
    def use_template(
        user_template_name: Templates,
        system_template_name: Templates = None,
        user_keyword="user",
        system_keyword="system",
        **kwargs,
    ) -> str | None:
        try:
            template_user = user_template_name.value
            template_system = (
                system_template_name.value if system_template_name else None
            )
            utemplate = parse_prompt(template_user, **kwargs)
            stemplate = (
                parse_prompt(template_system, **kwargs) if template_system else None
            )
            prompt = f"I will give you 'rules' 'content' two tags. You need to follow the rules!  <content>{utemplate}</content> <rules>{stemplate}</rules>"
            prompt = (
                anthropic.HUMAN_PROMPT
                + " "
                + prompt
                + " "
                + anthropic.AI_PROMPT
                + " "
                + '{"'
            )
            _LOGGER.info(f"使用模板成功")
            _LOGGER.debug(f"生成的prompt为：{prompt}")
            return prompt
        except Exception as e:
            _LOGGER.error(f"使用模板失败：{e}")
            traceback.print_exc()
            return None
