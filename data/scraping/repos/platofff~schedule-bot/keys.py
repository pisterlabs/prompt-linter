import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from os import environ
from time import time
from typing import Tuple, Awaitable

import aiofiles
import redis.asyncio as redis
import openai

from src.misc.redis_pool import redis_pool

logger = logging.getLogger('openai_keys')

class OpenAIKeysManager:
    _REDIS_KEY = 'openai_keys'
    _DELAY_SEC = 60

    @staticmethod
    def _check_key(key: str) -> Tuple[bool, str]:
        model_list: list = openai.Model.list(api_key=key).data
        model = next((x for x in model_list if x['id'] == environ['OPENAI_MODEL']), None)
        has_model = model is not None

        return has_model, key


    @staticmethod
    async def load_keys():
        async with redis.Redis(connection_pool=redis_pool) as conn:
            await conn.delete(OpenAIKeysManager._REDIS_KEY)

            async with aiofiles.open(environ['OPENAI_KEYS_FILE'], 'r') as f:
                keys = list(filter(bool, (await f.read()).split('\n')))

            result = []
            loop = asyncio.get_event_loop()

            with ThreadPoolExecutor() as executor:
                tasks = [loop.run_in_executor(executor, OpenAIKeysManager._check_key, key) for key in keys]

                for i, future in enumerate(asyncio.as_completed(tasks)):
                    is_available, key = await future
                    if is_available:
                        logger.info(f'Checked OpenAI keys {i + 1} of {len(keys)}')
                        await conn.zadd(OpenAIKeysManager._REDIS_KEY, {key: 0})
                    else:
                        logger.error(f'{environ["OPENAI_MODEL"]} is unavailable using API key {i + 1}')

        return result

    @staticmethod
    async def get_key() -> str:
        async with redis.Redis(connection_pool=redis_pool) as conn:
            while True:
                available = await conn.zrangebyscore(OpenAIKeysManager._REDIS_KEY, min=0, max=time(), num=1, start=0)
                if not available:
                    target = (await conn.zrangebyscore(OpenAIKeysManager._REDIS_KEY, min=0, max='+inf', num=1,
                                                              withscores=True, start=0))[0][1]
                    await asyncio.sleep(max(target - time() + 0.1, 0))
                    continue
                await conn.zadd(OpenAIKeysManager._REDIS_KEY, {available[0]: time()})
                return available[0].decode()

    @staticmethod
    async def delay_key(key: str) -> Awaitable:
        async with redis.Redis(connection_pool=redis_pool) as conn:
            return conn.zadd(OpenAIKeysManager._REDIS_KEY, {key: time() + OpenAIKeysManager._DELAY_SEC})
