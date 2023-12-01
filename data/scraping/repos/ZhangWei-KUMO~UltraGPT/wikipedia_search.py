from lanarky.responses import StreamingResponse
from langchain import WikipediaAPIWrapper
from agent_backend.schemas import ModelSettings
from agent_backend.web.api.agent.tools.tool import Tool
from agent_backend.web.api.agent.tools.utils import summarize
import wikipedia


class Wikipedia(Tool):
    description = (
        "搜索维基百科以获取有关历史人物、公司、事件、地点或研究的信息。这应该用于搜索特定名词的广泛概述。参数应该是一个简单的名词查询。"
    )
    public_description = "搜索维基百科以获取历史信息。"

    def __init__(self, model_settings: ModelSettings):
        super().__init__(model_settings)
        self.wikipedia = WikipediaAPIWrapper(
            wiki_client=None,  # Meta private value but mypy will complain its missing
        )

    async def call(self, goal: str, task: str, input_str: str) -> StreamingResponse:
        wikipedia.set_lang("zh")
        # 只有gpt4可以这么造
        res =  wikipedia.page(input_str)
        return res.content