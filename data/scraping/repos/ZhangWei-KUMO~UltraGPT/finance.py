from lanarky.responses import StreamingResponse
from langchain import WikipediaAPIWrapper
from agent_backend.schemas import ModelSettings
from agent_backend.web.api.agent.tools.tool import Tool
from agent_backend.web.api.agent.tools.utils import summarize
import os
# import tushare as ts
# from dotenv import load_dotenv
# load_dotenv()
# TUSHARE_API_KEY = os.getenv('TUSHARE_API_KEY')
# pro = ts.pro_api(TUSHARE_API_KEY)

class Finance(Tool):
    description = (
        "基于Tushare的数据，搜索股票信息。"
    )
    public_description = "基于Tushare的数据，搜索股票信息。"

    def __init__(self, model_settings: ModelSettings):
        super().__init__(model_settings)

    async def call(self, goal: str, task: str, input_str: str) -> StreamingResponse:
        # df = pro.balancesheet(ts_code='600000.SH', start_date='20180101', end_date='20230609')
        # return summarize(self.model_settings, goal, task, df[:6])
        return ""
