import openai
import nonebot

from .config import Config

plugin_config = Config.parse_obj(nonebot.get_driver().config.dict())


class Interviewer:
    def __init__(self, api_key, model_id, company, job):
        self.api_key = api_key
        self.model_id = model_id
        self.content = []
        self.company = company
        self.job = job

    async def get_ans(self, user_content):
        self.content.append({"role": "user", "content": user_content})
        res_ = await self.openai_interface(self.content)
        res = res_.choices[0].message.content
        while res.startswith("\n") != res.startswith("？"):
            res = res[1:]
        self.content.append({"role": 'assistant', "content": res})

        return res

    async def openai_interface(self, msg):
        openai.api_key = self.api_key
        if plugin_config.openai_api_base:
            openai.api_base = plugin_config.openai_api_base
        if plugin_config.openai_http_proxy:
            openai.proxy = {'http': plugin_config.openai_http_proxy, 'https': plugin_config.openai_http_proxy}

        response = await openai.ChatCompletion.acreate(
            model=self.model_id,
            messages=msg,
            temperature=0.2
        )

        return response

    async def init_interview(self):
        prompt = f"你是一名熟悉{self.company}的面试官。用户将成为候选人，您将向用户询问{self.job}职位的面试问题。希望你只作为面试官回答。不要一次写出所有的问题。希望你只对用户进行面试。问用户问题，等待用户的回答。不要写解释。你需要像面试官一样一个一个问题问用户，等用户回答。 若用户回答不上某个问题，那就继续问下一个问题 "
        self.content = [{"role": "system", "content": prompt}, {"role": "user", "content": "面试官你好"}]

        res_ = await self.openai_interface(self.content)
        res = res_.choices[0].message.content
        while res.startswith("\n") != res.startswith("？"):
            res = res[1:]

        return res
