from __future__ import annotations

from typing import Dict, Type

from ghoshell.ghost import Context, Thought, Operator
from ghoshell.llms import OpenAIFuncSchema
from ghoshell.llms.thinks import AgentThought
from ghoshell.llms.thinks import LLMFunc
from ghoshell.prototypes.playground.sphero.sphero_commands import SpheroCommand, Say
from ghoshell.prototypes.playground.sphero.sphero_ghost_core import SpheroGhostCore
from ghoshell.prototypes.playground.sphero.sphero_messages import SpheroCommandMessage


class SpheroLLMFunc(LLMFunc):

    def __init__(self, core: SpheroGhostCore, cmd: Type[SpheroCommand]):
        self.cmd: Type[SpheroCommand] = cmd
        self.core = core

    def name(self) -> str:
        return self.cmd.method

    def schema(self, ctx: Context, this: AgentThought) -> OpenAIFuncSchema:
        return OpenAIFuncSchema(
            name=self.cmd.method,
            desc=self.cmd.desc(),
            parameters_schema=self.cmd.model_json_schema(),
        )

    def call(self, ctx: Context, this: Thought, content: str, arguments: Dict | str | None) -> Operator | str | None:
        message = SpheroCommandMessage(runtime_mode=True)
        if content:
            message.add(Say(content=content))

        wrapped: SpheroCommand = self.wrap(self.cmd, arguments)
        message.add(wrapped)
        commands, ok = self.core.filter_commands_data(ctx, message.commands)
        if ok:
            message.commands = commands
        else:
            message.commands = []
            invalid = self.core.invalid_order()
            message.add(Say(content=invalid))

        ctx.send_at(this).output(message)
        return ctx.mind(this).awaits()
