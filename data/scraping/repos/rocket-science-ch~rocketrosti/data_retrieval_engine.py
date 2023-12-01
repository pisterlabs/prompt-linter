# Copyright (c) 2023 Rocket Science AG, Switzerland

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Allow users to ask questions about data. Answer those questions by delegating them to an LLM,
which in turn can execute specific functions like rtfm to answer them.
"""

from __future__ import annotations

import asyncio
import sys

import marshmallow as ma
from overrides import override

from rrosti.chat import chat_session
from rrosti.chat.state_machine import execution, interpolable
from rrosti.chat.state_machine.execution import load_and_run
from rrosti.llm_api import openai_api
from rrosti.servers.websocket_query_server import Frontend, QueryEngineBase


class DataQueryEngine(QueryEngineBase):
    _llm: chat_session.LLM
    _openai_provider: openai_api.OpenAIApiProvider

    def __init__(self, llm: chat_session.LLM, openai_provider: openai_api.OpenAIApiProvider) -> None:
        self._llm = llm
        self._openai_provider = openai_provider

    @override
    async def ensure_loaded(self) -> None:
        await asyncio.gather(interpolable.ensure_loaded(), execution.ensure_loaded())

    @override
    async def arun(self, frontend: Frontend) -> None:
        try:
            await load_and_run(llm=self._llm, frontend=frontend, openai_provider=self._openai_provider)
        except ma.exceptions.ValidationError as e:
            print(e.messages, sys.stderr)
            raise
