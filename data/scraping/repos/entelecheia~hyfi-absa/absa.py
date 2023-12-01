import json
import logging

from hyabsa import HyFI
from hyabsa.llms import OpenAIChatCompletion

from .base import BaseAgent
from .results import AgentResult

logger = logging.getLogger(__name__)


class AbsaAgent(BaseAgent):
    _config_name_: str = "absa"
    llm: OpenAIChatCompletion = OpenAIChatCompletion()

    def execute(
        self,
        id: str,
        text: str,
    ) -> str:
        msg = self.build_message(text)
        rst = self.llm.request(msg)
        result = AgentResult.from_chat_reponse(id, rst)
        if self.output_filepath:
            HyFI.append_to_jsonl(result.model_dump(), self.output_filepath)
        return json.dumps(result.model_dump(), ensure_ascii=False)
