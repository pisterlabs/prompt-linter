from typing import Any, Dict, List, Optional

from langchain.agents.agent import AgentExecutor
from langchain.agents.tools import Tool
from langchain.chains.base import Chain
from langchain.llms.base import BaseLLM
from langchain.schema import AgentAction, AgentFinish
from langchain_contrib.tools.terminal import SafeTerminalChain, TerminalToolChain

from zamm.chains.bash_action_prompt import (
    EMPLOYEE_TEACHING_INTRO_TEMPLATE,
    FOLLOW_INSTRUCTIONS_TEMPLATE,
)
from zamm.utils import f_join

from .base import CustomAgent
from .employee_actions import action_with_thought_chain, default_action_chain
from .memory import AgentMemory, BaseAgentMemory
from .step import StepOutput


class ZammEmployeeBrain(CustomAgent):
    condense_memory: bool = False

    @property
    def _agent_type(self) -> str:
        """Return Identifier of agent type."""
        return "zamm-employee"

    def _get_next_action(self, full_inputs: Dict[str, str]) -> AgentAction:
        raise NotImplementedError()

    def _construct_scratchpad_base(
        self, memory: BaseAgentMemory, condensed: bool
    ) -> str:
        logs = []
        steps = memory.steps()
        for i, output in enumerate(steps):
            previous = steps[i - 1] if i > 0 else None
            next = steps[i + 1] if i < len(steps) - 1 else None
            logs.append(output.log(previous=previous, next=next, condensed=condensed))
        scratch = f_join("\n", logs)
        if scratch == "":
            return scratch
        return "\n" + scratch + "\n"

    def _construct_scratchpad_structured(self, memory: BaseAgentMemory) -> str:
        return self._construct_scratchpad_base(memory, condensed=self.condense_memory)

    def _construct_scratchpad_final(self, memory: BaseAgentMemory) -> str:
        return self._construct_scratchpad_base(memory, condensed=False)


class ZammEmployee(AgentExecutor):
    agent: ZammEmployeeBrain
    terminal_chain: Chain
    max_iterations: Optional[int] = 100
    think_before_acting: bool = False

    def __init__(
        self,
        llm: BaseLLM,
        condense_memory: bool = False,
        tools: Optional[List[Tool]] = None,
        terminal_safe_mode: bool = True,
        **kwargs,
    ):
        tools = tools if tools else []
        brain = ZammEmployeeBrain.from_llm(llm)
        brain.condense_memory = condense_memory
        super().__init__(
            agent=brain,
            tools=tools,
            terminal_chain=SafeTerminalChain()
            if terminal_safe_mode
            else TerminalToolChain(),
            **kwargs,
        )

    @property
    def input_keys(self) -> List[str]:
        return ["task", "documentation"]

    def _custom_take_next_step(self, memory: BaseAgentMemory) -> StepOutput:
        if "documentation" in memory.inputs:
            prefix = FOLLOW_INSTRUCTIONS_TEMPLATE
        else:
            prefix = EMPLOYEE_TEACHING_INTRO_TEMPLATE

        assert isinstance(self.agent.llm_chain.llm, BaseLLM)
        llm: BaseLLM = self.agent.llm_chain.llm

        def create_new_employee() -> AgentExecutor:
            return self.__class__(
                llm=llm,
                condense_memory=self.agent.condense_memory,
                terminal_safe_mode=isinstance(self.terminal_chain, SafeTerminalChain),
            )

        if self.think_before_acting:
            action_chain = action_with_thought_chain(
                llm=llm,
                prefix=prefix,
                terminal_chain=self.terminal_chain,
                agent_creator=create_new_employee,
            )
            output_key = action_chain.chains[-1].step_output_key
        else:
            action_chain = default_action_chain(
                llm=llm,
                prefix=prefix,
                terminal_chain=self.terminal_chain,
                agent_creator=create_new_employee,
            )
            output_key = action_chain.step_output_key
        results = action_chain(
            {
                **memory.inputs,
                "agent_scratchpad": self.agent._construct_scratchpad_structured(memory),
            }
        )
        return results[output_key]

    def _return_structured(
        self, output: StepOutput, memory: AgentMemory
    ) -> Dict[str, Any]:
        memory.add_step(output)
        task = memory.inputs["task"]
        logs = self.agent._construct_scratchpad_final(memory)
        tutorial = f"""
Say you want to do the following task:

> {task}

You can do so by following these steps:

{logs}
""".strip()
        return {"output": tutorial}

    def _validate_inputs(self, inputs: Dict[str, str]) -> None:
        """Skip validation because some inputs are optional."""

    def _call(self, inputs: Dict[str, str]) -> Dict[str, Any]:
        """Stripped down version of AgentExecutor._call."""
        # Do any preparation necessary when receiving a new input.
        self.agent.prepare_for_new_call()

        memory = AgentMemory(inputs=inputs)
        iterations = 0
        # We now enter the agent loop (until it returns something).
        while self._should_continue(iterations):
            step_result = self._custom_take_next_step(memory)
            # If the tool chosen is the finishing tool, then we end and return.
            if isinstance(step_result.decision, AgentFinish):
                return self._return_structured(step_result, memory)

            memory.add_step(step_result)
            iterations += 1
        output = self.agent.return_stopped_response(
            self.early_stopping_method, memory.as_intermediate_steps(), **inputs
        )
        return self._return(output, memory.as_intermediate_steps())
