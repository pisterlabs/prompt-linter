"""The OpenAI Conversation integration."""

import os
import logging
import asyncio
from kani import Kani, chat_in_terminal_async
from kani.engines.openai import OpenAIEngine

from abilities.base import BaseAbility
from abilities.homeassistant import HomeAssistantAbility
from abilities.google import GoogleAbility

logging.basicConfig(level=logging.DEBUG)
_LOGGER = logging.getLogger(__name__)

async def get_ai(openai_key: str, abilities: [BaseAbility] = []):
    _LOGGER.debug('Starting up OpenAIEngine')
    engine = OpenAIEngine(openai_key, model="gpt-3.5-turbo-0613", max_context_size=4096)

    system_prompt = ''
    chat_history = []
    all_functions = []

    _LOGGER.debug(f'Registering abilities')
    for ability in abilities:
        _LOGGER.debug(f'\nRegistering ability: {ability.name}')
        prompt = ability.sys_prompt()
        history = await ability.chat_history()
        functions = ability.registered_functions()

        _LOGGER.debug(f'System prompt: {prompt}')
        _LOGGER.debug(f'Chat history: {history}')
        _LOGGER.debug(f'Functions: {functions}')

        system_prompt += prompt
        chat_history += history
        all_functions += functions

    return Kani(
        engine=engine,
        system_prompt=system_prompt,
        chat_history=chat_history,
        functions=all_functions,
    )

async def main_development():
    abilities = [
        HomeAssistantAbility(api_key=os.getenv('HOMEASSISTANT_KEY'), base_url=os.getenv('HOMEASSISTANT_URL')),
        GoogleAbility(api_key=os.getenv('GOOGLE_API_KEY'), cx_key=os.getenv('GOOGLE_CX_KEY')),
    ]
    openai_key = os.getenv('OPENAI_KEY')
    ai = await get_ai(openai_key=openai_key, abilities=abilities)

    await chat_in_terminal_async(ai)


if __name__ == '__main__':
    asyncio.run(main_development())
