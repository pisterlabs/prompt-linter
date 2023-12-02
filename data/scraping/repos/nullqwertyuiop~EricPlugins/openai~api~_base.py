import asyncio
import random

from creart import it
from graia.saya import Channel
from kayaku import create

from library.model import EricConfig
from library.util.session_container import SessionContainer
from module.openai.config import OpenAIConfig

channel = Channel.current()


class OpenAIAPIBase:
    BASE: str = "https://api.openai.com/v1"
    OBJECT: str

    @property
    def headers(self) -> dict[str, str]:
        cfg: OpenAIConfig = create(OpenAIConfig)
        if not cfg.api_keys:
            raise ValueError("OpenAI API Key 未配置")
        api_key: str = random.choice(cfg.api_keys)
        return {"ContentType": "application/json", "Authorization": f"Bearer {api_key}"}

    @property
    def proxy(self) -> str:
        return create(EricConfig).proxy

    @property
    def url(self) -> str:
        return f"{self.BASE}/" + self.OBJECT.replace(".", "/")

    async def _call_impl(self, /, **kwargs) -> dict:
        session = await it(SessionContainer).get(channel.module)
        async with session.post(
                self.url, headers=self.headers, json=kwargs, proxy=self.proxy
        ) as resp:
            return await resp.json()

    async def _call(self, /, timeout: int = 30, **kwargs) -> dict:
        return await asyncio.wait_for(self._call_impl(**kwargs), timeout=timeout)
