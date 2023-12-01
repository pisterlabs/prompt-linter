from __future__ import annotations

from typing import Any, Callable, List, NamedTuple, Optional, Sequence
from langchain.agents.agent import BaseMultiActionAgent, BaseSingleActionAgent
from pydantic import Field
from chemagent.agents.prompts import *
from langchain.agents.agent import Agent, AgentExecutor, AgentOutputParser
from langchain.agents.agent_types import AgentType
from langchain.agents.mrkl.output_parser import MRKLOutputParser
# from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS, PREFIX, SUFFIX
from langchain.agents.tools import Tool
from langchain.agents.utils import validate_tools_single_input
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools.base import BaseTool
from ..tools.databases import strip_non_alphanumeric

import asyncio
import json
import logging
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import yaml
from pydantic import BaseModel, root_validator

from langchain.agents.agent_iterator import AgentExecutorIterator
from langchain.agents.agent_types import AgentType
from langchain.agents.tools import InvalidTool
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    AsyncCallbackManagerForToolRun,
    CallbackManagerForChainRun,
    CallbackManagerForToolRun,
    Callbacks,
)
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import (
    AgentAction,
    AgentFinish,
    BaseOutputParser,
    BasePromptTemplate,
    OutputParserException,
)
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import BaseMessage
from langchain.tools.base import BaseTool
from langchain.utilities.asyncio import asyncio_timeout
from langchain.utils.input import get_color_mapping

from dataclasses import dataclass
from typing import NamedTuple, Union

@dataclass
class AgentAnalysis():
    analysis: str
    

class AgentAnalysisOutputParser(AgentOutputParser):
    def parse(self, text: str) -> AgentAnalysis:
        return AgentAnalysis(analysis=text)

class FewShotAgent(BaseModel):
    output_parser: AgentOutputParser = Field(default_factory=MRKLOutputParser)
    output_parser_agent_analysis: AgentOutputParser = Field(default_factory=AgentAnalysisOutputParser)
    llm_chain_dict: dict = Field(default_factory=dict)
    llm_prefix: List[str] = Field(default_factory=list)
    allowed_tools: Optional[List[str]] = None
    return_values: List[str] = ["output"]

    # # TODO: is this enough?????? --- check https://github.com/pydantic/pydantic/blob/586ed21dba67d1024eb2b4fea362a0bdfb87ba43/pydantic/v1/main.py#L310
    # def __init__(cls, 
    #              llm_chain_dict: dict,
    #              llm_prefix: List[str],
    #             allowed_tools: Optional[List[str]] = None,
    #             output_parser: Optional[AgentOutputParser] = None,
    #             output_parser_agent_analysis: Optional[AgentOutputParser] = None,
    #              **kwargs: Any):
    #     cls.llm_chain_list = llm_chain_dict
    #     cls.llm_prefix = llm_prefix
    #     cls.allowed_tools = allowed_tools
    #     cls.output_parser = output_parser or cls._get_default_output_parser()
    #     cls.output_parser_agent_analysis = output_parser_agent_analysis or cls._get_default_output_parser_agent_analysis()


    @property
    def _stop(self) -> List[str]:
        return [f"\n{self.observation_prefix}"]
    
    @classmethod
    def _get_default_output_parser(cls, **kwargs: Any) -> AgentOutputParser:
        return MRKLOutputParser()
    
    @classmethod
    def _get_default_output_parser_agent_analysis(cls, **kwargs: Any) -> AgentOutputParser:
        return AgentAnalysisOutputParser()
    
    @property
    def _agent_type(self) -> str:
        """Return Identifier of agent type."""
        return AgentType.ZERO_SHOT_REACT_DESCRIPTION

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "Observation: "

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the llm call with."""
        return "Thought:"

    @classmethod
    def create_prompt(
        cls,
        tools: Sequence[BaseTool],
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        input_variables: Optional[List[str]] = None,
    ) -> PromptTemplate:
        """Create prompt in the style of the zero shot agent.

        Args:
            tools: List of tools the agent will have access to, used to format the
                prompt.
            prefix: String to put before the list of tools.
            suffix: String to put after the list of tools.
            input_variables: List of input variables the final prompt will expect.

        Returns:
            A PromptTemplate with the template assembled from the pieces here.
        """
        tool_strings = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
        tool_names = ", ".join([tool.name for tool in tools])
        format_instructions = format_instructions.format(tool_names=tool_names)
        template = "\n\n".join([prefix, tool_strings, format_instructions, suffix])
        if input_variables is None:
            input_variables = ["input", "agent_scratchpad"]
        return PromptTemplate(template=template, input_variables=input_variables)
    
    @classmethod
    def create_fewshot_prompt(
        cls,
        examples: str = EXAMPLE_FEW_SHOT_CHEM, #里面带input的
        prefix: str = PREFIX_FEW_SHOT_CHEM,
        suffix: str = SUFFIX_FEW_SHOT_CHEM,
        input_variables: Optional[List[str]] = None,
    ) -> PromptTemplate:

        if input_variables is None:
            input_variables = ["input", "examples"]
        template = "\n\n".join([prefix, examples, suffix])
        return PromptTemplate(template=template, input_variables=input_variables)
        

    @classmethod
    def from_llm_and_tools(
        cls,
        llms: List[BaseLanguageModel], 
        tools: Sequence[BaseTool],
        prompts_list : List[PromptTemplate] = None,
        llms_prefix: List[str] = ["agent", "fewshot_analysis"],
        num_llm: int = 2,
        callback_manager: Optional[BaseCallbackManager] = None,
        output_parser: Optional[AgentOutputParser] = None,
        prefix: str = PREFIX, # 这里的prefix，suffix，format_instructions是主agent的
        suffix: str = SUFFIX,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        input_variables: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Agent:
        assert len(llms) == num_llm
        """Construct an agent from an LLM and tools."""
        cls._validate_tools(tools)
        prompt_main_agent = cls.create_prompt(
            tools,
            prefix=prefix,
            suffix=suffix,
            format_instructions=format_instructions,
            input_variables=input_variables,
        )
        
        if prompts_list is None:
            prompt_main_agent = cls.create_prompt(
                tools,
                prefix=prefix,
                suffix=suffix,
                format_instructions=format_instructions,
                input_variables=input_variables,
            )
            prompts_list = [prompt_main_agent]
            for i in range(num_llm-1):
                prompts_list.append(cls.create_fewshot_prompt()) 

        llm_chain_dict = {}
        for i in range(num_llm):
            llm_chain = LLMChain(
                llm=llms[i],
                prompt=prompts_list[i],
                callback_manager=callback_manager,
            )
            llm_chain_dict[llms_prefix[i]] = llm_chain
        
        tool_names = [tool.name for tool in tools]
        _output_parser = output_parser or cls._get_default_output_parser()
        # TODO: edit cls initialize (重写__init__方法) --------------------------------------------------------------
        return cls(
            llm_chain_dict=llm_chain_dict,
            llm_prefix=llms_prefix,
            allowed_tools=tool_names,
            output_parser=_output_parser,
            output_parser_agent_analysis= cls._get_default_output_parser_agent_analysis(),
            **kwargs,
        )
         # ---------------------------------------------------------------------------------------------------------------
    
    # TODO: AttributeError: 'FewShotAgent' object has no attribute 'stop'??
    @property
    def stop(self):
        return [f"\n{self.observation_prefix}"]
    
    # TODO: input keys
    @property
    def input_keys(self) -> List[str]:
        """Return the input keys.

        :meta private:
        """
        return list(set(self.llm_chain_dict['agent'].input_keys) - {"agent_scratchpad"})

    # TODO ----------------------------------------------------------------------------------------------
        
    def get_allowed_tools(self) -> Optional[List[str]]:
        return None
    
    def _get_next_action(self, full_inputs: Dict[str, str]) -> AgentAction:
        full_output = self.llm_chain_dict['agent'].predict(**full_inputs)
        parsed_output = self._extract_tool_and_input(full_output)
        while parsed_output is None:
            full_output = self._fix_text(full_output)
            full_inputs["agent_scratchpad"] += full_output
            output = self.llm_chain_dict['agent'].predict(**full_inputs)
            full_output += output
            parsed_output = self._extract_tool_and_input(full_output)
        return AgentAction(
            tool=parsed_output[0], tool_input=parsed_output[1], log=full_output
        )

    async def _aget_next_action(self, full_inputs: Dict[str, str]) -> AgentAction:
        full_output = await self.llm_chain_dict['agent'].apredict(**full_inputs)
        parsed_output = self._extract_tool_and_input(full_output)
        while parsed_output is None:
            full_output = self._fix_text(full_output)
            full_inputs["agent_scratchpad"] += full_output
            output = await self.llm_chain_dict['agent'].apredict(**full_inputs)
            full_output += output
            parsed_output = self._extract_tool_and_input(full_output)
        return AgentAction(
            tool=parsed_output[0], tool_input=parsed_output[1], log=full_output
        )
    
    # TODO ----------------------------------------------------------------------------------------------

    def tool_run_logging_kwargs(self) -> Dict:
        return {
            "llm_prefix": "",
            "observation_prefix": "" if len(self.stop) == 0 else self.stop[0],
        }


    def _construct_scratchpad(
        self, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> Union[str, List[BaseMessage]]:
        """Construct the scratchpad that lets the agent continue its thought process."""
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\n{self.observation_prefix}{observation}\n{self.llm_prefix}"
        return thoughts

    def get_full_inputs(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Dict[str, Any]:
        """Create the full inputs for the LLMChain from intermediate steps."""
        thoughts = self._construct_scratchpad(intermediate_steps)
        new_inputs = {"agent_scratchpad": thoughts, "stop": self._stop}
        full_inputs = {**kwargs, **new_inputs}
        return full_inputs
    
    
    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],  # AgentAction是一个类包括tool，tool_input还有log；intermediate_steps就是形如[(agentaction, observation)]的列表
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """

        # llm_chain.predict()在文档中：Format prompt with kwargs and pass to LLM.
        full_inputs = self.get_full_inputs(intermediate_steps, **kwargs) # 把GPT上一轮的思考
        full_output = self.llm_chain_dict['agent'].predict(callbacks=callbacks, **full_inputs)
        return self.output_parser.parse(full_output)

    async def aplan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        full_inputs = self.get_full_inputs(intermediate_steps, **kwargs) # 把GPT上一轮的思考
        full_output = await self.llm_chain_dict['agent'].apredict(callbacks=callbacks, **full_inputs)
        return self.output_parser.parse(full_output)
    

    def get_fewshot_analysis(
        self,
        examples: str,
        **kwargs: Any,
    ) -> AgentAnalysis:
        inputs = {'examples': examples, **kwargs}
        output = self.llm_chain_dict['fewshot_analysis'].predict(**inputs) # input_variables = ["input", "examples"]
        return self.output_parser_agent_analysis.parse(output)

    async def aget_fewshot_analysis(
        self,
        **kwargs: Any,
    ) -> AgentAnalysis:
        output = await self.llm_chain_dict['fewshot_analysis'].apredict(**kwargs)
        return self.output_parser_agent_analysis.parse(output)

    @classmethod
    def _validate_tools(cls, tools: Sequence[BaseTool]) -> None:
        validate_tools_single_input(cls.__name__, tools)
        if len(tools) == 0:
            raise ValueError(
                f"Got no tools for {cls.__name__}. At least one tool must be provided."
            )
        for tool in tools:
            if tool.description is None:
                raise ValueError(
                    f"Got a tool {tool.name} without a description. For this agent, "
                    f"a description must always be provided."
                )
        # super()._validate_tools(tools)

    def return_stopped_response(
        self,
        early_stopping_method: str,
        intermediate_steps: List[Tuple[AgentAction, str]],
        **kwargs: Any,
    ) -> AgentFinish:
        """Return response when agent has been stopped due to max iterations."""
        if early_stopping_method == "force":
            # `force` just returns a constant string
            return AgentFinish({"output": "Agent stopped due to max iterations."}, "")
        elif early_stopping_method == "generate":
            # Generate does one final forward pass
            thoughts = ""
            for action, observation in intermediate_steps:
                thoughts += action.log
                thoughts += (
                    f"\n{self.observation_prefix}{observation}\n{self.llm_prefix}"
                )
            # Adding to the previous steps, we now tell the LLM to make a final pred
            thoughts += (
                "\n\nI now need to return a final answer based on the previous steps:"
            )
            new_inputs = {"agent_scratchpad": thoughts, "stop": self._stop}
            full_inputs = {**kwargs, **new_inputs}
            full_output = self.llm_chain.predict(**full_inputs)
            # We try to extract a final answer
            parsed_output = self._extract_tool_and_input(full_output)
            if parsed_output is None:
                # If we cannot extract, we just return the full output
                return AgentFinish({"output": full_output}, full_output)
            tool, tool_input = parsed_output
            if tool == self.finish_tool_name:
                # If we can extract, we send the correct stuff
                return AgentFinish({"output": tool_input}, full_output)
            else:
                # If we can extract, but the tool is not the final tool,
                # we just return the full output
                return AgentFinish({"output": full_output}, full_output)
        else:
            raise ValueError(
                "early_stopping_method should be one of `force` or `generate`, "
                f"got {early_stopping_method}"
            )


class FewShotAgentExecutor(Chain):
    """Agent that is using tools."""

    agent: FewShotAgent
    """The agent to run for creating a plan and determining actions
    to take at each step of the execution loop."""
    tools: Sequence[BaseTool]
    """The valid tools the agent can call."""
    return_intermediate_steps: bool = False
    """Whether to return the agent's trajectory of intermediate steps
    at the end in addition to the final output."""
    max_iterations: Optional[int] = 15
    """The maximum number of steps to take before ending the execution
    loop.
    
    Setting to 'None' could lead to an infinite loop."""
    max_execution_time: Optional[float] = None
    """The maximum amount of wall clock time to spend in the execution
    loop.
    """
    early_stopping_method: str = "force"
    """The method to use for early stopping if the agent never
    returns `AgentFinish`. Either 'force' or 'generate'.

    `"force"` returns a string saying that it stopped because it met a
        time or iteration limit.
    
    `"generate"` calls the agent's LLM Chain one final time to generate
        a final answer based on the previous steps.
    """
    handle_parsing_errors: Union[
        bool, str, Callable[[OutputParserException], str]
    ] = False
    """How to handle errors raised by the agent's output parser.
    Defaults to `False`, which raises the error.
s
    If `true`, the error will be sent back to the LLM as an observation.
    If a string, the string itself will be sent to the LLM as an observation.
    If a callable function, the function will be called with the exception
     as an argument, and the result of that function will be passed to the agent
      as an observation.
    """
    trim_intermediate_steps: Union[
        int, Callable[[List[Tuple[AgentAction, str]]], List[Tuple[AgentAction, str]]]
    ] = -1

    @classmethod
    def from_agent_and_tools(
        cls,
        agent: FewShotAgent,
        tools: Sequence[BaseTool],
        callback_manager: Optional[BaseCallbackManager] = None,
        **kwargs: Any,
    ) -> AgentExecutor:
        """Create from agent and tools."""
        return cls(
            agent=agent, tools=tools, callback_manager=callback_manager, **kwargs
        )

    
    @root_validator()
    def validate_tools(cls, values: Dict) -> Dict:
        """Validate that tools are compatible with agent."""
        agent = values["agent"]
        tools = values["tools"]
        allowed_tools = agent.get_allowed_tools()
        if allowed_tools is not None:
            if set(allowed_tools) != set([tool.name for tool in tools]):
                raise ValueError(
                    f"Allowed tools ({allowed_tools}) different than "
                    f"provided tools ({[tool.name for tool in tools]})"
                )
        return values

    @root_validator()
    def validate_return_direct_tool(cls, values: Dict) -> Dict:
        """Validate that tools are compatible with agent."""
        agent = values["agent"]
        tools = values["tools"]
        if isinstance(agent, BaseMultiActionAgent):
            for tool in tools:
                if tool.return_direct:
                    raise ValueError(
                        "Tools that have `return_direct=True` are not allowed "
                        "in multi-action agents"
                    )
        return values

    def save(self, file_path: Union[Path, str]) -> None:
        """Raise error - saving not supported for Agent Executors."""
        raise ValueError(
            "Saving not supported for agent executors. "
            "If you are trying to save the agent, please use the "
            "`.save_agent(...)`"
        )

    def save_agent(self, file_path: Union[Path, str]) -> None:
        """Save the underlying agent."""
        return self.agent.save(file_path)

    def iter(
        self,
        inputs: Any,
        callbacks: Callbacks = None,
        *,
        include_run_info: bool = False,
        async_: bool = False,
    ) -> AgentExecutorIterator:
        """Enables iteration over steps taken to reach final output."""
        return AgentExecutorIterator(
            self,
            inputs,
            callbacks,
            tags=self.tags,
            include_run_info=include_run_info,
            async_=async_,
        )

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys.

        :meta private:
        """
        return self.agent.input_keys

    @property
    def output_keys(self) -> List[str]:
        """Return the singular output key.

        :meta private:
        """
        if self.return_intermediate_steps:
            return self.agent.return_values + ["intermediate_steps"]
        else:
            return self.agent.return_values

    def lookup_tool(self, name: str) -> BaseTool:
        """Lookup tool by name."""
        return {tool.name: tool for tool in self.tools}[name]

    def _should_continue(self, iterations: int, time_elapsed: float) -> bool:
        if self.max_iterations is not None and iterations >= self.max_iterations:
            return False
        if (
            self.max_execution_time is not None
            and time_elapsed >= self.max_execution_time
        ):
            return False

        return True

    def _return(
        self,
        output: AgentFinish,
        intermediate_steps: list,
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        if run_manager:
            run_manager.on_agent_finish(output, color="green", verbose=self.verbose)
        final_output = output.return_values
        if self.return_intermediate_steps:
            final_output["intermediate_steps"] = intermediate_steps
        return final_output

    async def _areturn(
        self,
        output: AgentFinish,
        intermediate_steps: list,
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        if run_manager:
            await run_manager.on_agent_finish(
                output, color="green", verbose=self.verbose
            )
        final_output = output.return_values
        if self.return_intermediate_steps:
            final_output["intermediate_steps"] = intermediate_steps
        return final_output

    def _take_next_step(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        inputs: Dict[str, str], # 这里就是传question？（Keys to pass to prompt template）是一个字典形式的参数，最后会传到self.llm_chain.predict(callbacks=callbacks, **full_inputs)的full_inputs里面
        intermediate_steps: List[Tuple[AgentAction, str]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Union[AgentFinish, List[Tuple[AgentAction, str]]]:
        """Take a single step in the thought-action-observation loop.

        Override this to take control of how the agent makes and acts on choices.
        """
        try:
            # 在默认情况下（trim_intermediate_steps=-1），_prepare_intermediate_steps返回intermediate_steps
            intermediate_steps = self._prepare_intermediate_steps(intermediate_steps)

            output = self.agent.plan( # 见148行，这个plan函数会返回“AgentAction”类或者“AgentFinish”类
                intermediate_steps,
                callbacks=run_manager.get_child() if run_manager else None,
                **inputs,
            )

        except OutputParserException as e:
            if isinstance(self.handle_parsing_errors, bool):
                raise_error = not self.handle_parsing_errors
            else:
                raise_error = False
            if raise_error:
                raise e
            text = str(e)
            if isinstance(self.handle_parsing_errors, bool):
                if e.send_to_llm:
                    observation = str(e.observation)
                    text = str(e.llm_output)
                else:
                    observation = "Invalid or incomplete response"
            elif isinstance(self.handle_parsing_errors, str):
                observation = self.handle_parsing_errors
            elif callable(self.handle_parsing_errors):
                observation = self.handle_parsing_errors(e)
            else:
                raise ValueError("Got unexpected type of `handle_parsing_errors`")
            output = AgentAction("_Exception", observation, text)
            if run_manager:
                run_manager.on_agent_action(output, color="green")
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            observation = ExceptionTool().run(
                output.tool_input,
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
            return [(output, observation)]
        # If the tool chosen is the finishing tool, then we end and return.
        if isinstance(output, AgentFinish):
            return output
        actions: List[AgentAction]
        if isinstance(output, AgentAction):
            actions = [output]
        else:
            actions = output
        result = []
        for agent_action in actions:
            if run_manager:
                run_manager.on_agent_action(agent_action, color="green")
            # Otherwise we lookup the tool
            if agent_action.tool in name_to_tool_map:
                tool = name_to_tool_map[agent_action.tool]
                return_direct = tool.return_direct
                color = color_mapping[agent_action.tool]
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                if return_direct:
                    tool_run_kwargs["llm_prefix"] = ""
                # We then call the tool on the tool input to get an observation
                observation = tool.run(
                    agent_action.tool_input,
                    verbose=self.verbose,
                    color=color,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **tool_run_kwargs,
                )
            else:
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                observation = InvalidTool().run(
                    agent_action.tool,
                    verbose=self.verbose,
                    color=None,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **tool_run_kwargs,
                )
            result.append((agent_action, observation))
        return result

    async def _atake_next_step(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        inputs: Dict[str, str],
        intermediate_steps: List[Tuple[AgentAction, str]],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Union[AgentFinish, List[Tuple[AgentAction, str]]]:
        """Take a single step in the thought-action-observation loop.

        Override this to take control of how the agent makes and acts on choices.
        """
        try:
            intermediate_steps = self._prepare_intermediate_steps(intermediate_steps)

            # Call the LLM to see what to do.
            output = await self.agent.aplan(
                intermediate_steps,
                callbacks=run_manager.get_child() if run_manager else None,
                **inputs,
            )
        except OutputParserException as e:
            if isinstance(self.handle_parsing_errors, bool):
                raise_error = not self.handle_parsing_errors
            else:
                raise_error = False
            if raise_error:
                raise e
            text = str(e)
            if isinstance(self.handle_parsing_errors, bool):
                if e.send_to_llm:
                    observation = str(e.observation)
                    text = str(e.llm_output)
                else:
                    observation = "Invalid or incomplete response"
            elif isinstance(self.handle_parsing_errors, str):
                observation = self.handle_parsing_errors
            elif callable(self.handle_parsing_errors):
                observation = self.handle_parsing_errors(e)
            else:
                raise ValueError("Got unexpected type of `handle_parsing_errors`")
            output = AgentAction("_Exception", observation, text)
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            observation = await ExceptionTool().arun(
                output.tool_input,
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
            return [(output, observation)]
        # If the tool chosen is the finishing tool, then we end and return.
        if isinstance(output, AgentFinish):
            return output
        actions: List[AgentAction]
        if isinstance(output, AgentAction):
            actions = [output]
        else:
            actions = output

        async def _aperform_agent_action(
            agent_action: AgentAction,
        ) -> Tuple[AgentAction, str]:
            if run_manager:
                await run_manager.on_agent_action(
                    agent_action, verbose=self.verbose, color="green"
                )
            # Otherwise we lookup the tool
            if agent_action.tool in name_to_tool_map:
                tool = name_to_tool_map[agent_action.tool]
                return_direct = tool.return_direct
                color = color_mapping[agent_action.tool]
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                if return_direct:
                    tool_run_kwargs["llm_prefix"] = ""
                # We then call the tool on the tool input to get an observation
                observation = await tool.arun(
                    agent_action.tool_input,
                    verbose=self.verbose,
                    color=color,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **tool_run_kwargs,
                )
            else:
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                observation = await InvalidTool().arun(
                    agent_action.tool,
                    verbose=self.verbose,
                    color=None,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **tool_run_kwargs,
                )
            return agent_action, observation

        # Use asyncio.gather to run multiple tool.arun() calls concurrently
        result = await asyncio.gather(
            *[_aperform_agent_action(agent_action) for agent_action in actions]
        )

        return list(result)

    def _call(
        self,
        inputs: Dict[str, str], # inputs就是调用agent时候的问题
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run text through and get agent response."""
        # Construct a mapping of tool name to tool for easy lookup
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        # We construct a mapping from each tool to a color, used for logging.
        color_mapping = get_color_mapping(
            [tool.name for tool in self.tools], excluded_colors=["green", "red"]
        )
        intermediate_steps: List[Tuple[AgentAction, str]] = []
        # Let's start tracking the number of iterations and time elapsed
        iterations = 0
        time_elapsed = 0.0
        start_time = time.time()
        # We now enter the agent loop (until it returns something).
        while self._should_continue(iterations, time_elapsed):
            next_step_output = self._take_next_step( # next_step_output返回AgentFinish或者[(output, observation)]，output就是AgentAction类(包括tool, tool_input, answe)
                name_to_tool_map,
                color_mapping,
                inputs, # 是一个字典形式的参数，最后会传到self.llm_chain.predict(callbacks=callbacks, **full_inputs)的full_inputs里面
                intermediate_steps,
                run_manager=run_manager, # None
            )

            # -------------------------QueryUSPTOWithType -> Different Prompts------------------------
            # 使用QueryUSPTOWithType之后使用自定义的prompts，而不是agent里面的prompts，来思考一次，并加入intermediate prompts
            if isinstance(next_step_output, List) and len(next_step_output) == 1 \
                and isinstance(next_step_output[0][0],AgentAction) \
              and next_step_output[0][0].tool == "QueryUSPTOWithType" and next_step_output[0][1].startswith('valid input'):
                
                examples = next_step_output[0][1].split('******')[-1]
                input_simles = strip_non_alphanumeric(next_step_output[0][0].tool_input).split(',')[0]
                get_fewshot_analysis_output = self.agent.get_fewshot_analysis(examples=examples, input = input_simles).analysis
                 ## 把ActionAnalysis的结果 转化为 Tuple[AgentAction, str] 加入到intermediate_steps里面 
                fewshot_step_output = (next_step_output[0][0], get_fewshot_analysis_output) # Tuple[AgentAction, str]
                intermediate_steps.append(fewshot_step_output)
                ## 把ActionAnalysis的结果 转化为 Tuple[AgentAction, str] 加入到intermediate_steps里面

                iterations += 1
                time_elapsed = time.time() - start_time
                continue
            # -----------------------------------------------------------------------------------------

            # 如果next_step_output是AgentFinish类，那么就返回了
            if isinstance(next_step_output, AgentFinish):
                return self._return(
                    next_step_output, intermediate_steps, run_manager=run_manager
                )

            # 如果next_step_output是一个List[Tuple[AgentAction, str]]
            # 那么就把它加到intermediate_steps里面
            intermediate_steps.extend(next_step_output)
            if len(next_step_output) == 1:
                next_step_action = next_step_output[0]
                # See if tool should return directly
                tool_return = self._get_tool_return(next_step_action)
                if tool_return is not None:
                    return self._return(
                        tool_return, intermediate_steps, run_manager=run_manager
                    )
            iterations += 1
            time_elapsed = time.time() - start_time

        ### WHILE LOOP END ###
        output = self.agent.return_stopped_response(
            self.early_stopping_method, intermediate_steps, **inputs
        )
        return self._return(output, intermediate_steps, run_manager=run_manager)

    async def _acall(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        """Run text through and get agent response."""
        # Construct a mapping of tool name to tool for easy lookup
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        # We construct a mapping from each tool to a color, used for logging.
        color_mapping = get_color_mapping(
            [tool.name for tool in self.tools], excluded_colors=["green"]
        )
        intermediate_steps: List[Tuple[AgentAction, str]] = []
        # Let's start tracking the number of iterations and time elapsed
        iterations = 0
        time_elapsed = 0.0
        start_time = time.time()
        # We now enter the agent loop (until it returns something).
        async with asyncio_timeout(self.max_execution_time):
            try:
                while self._should_continue(iterations, time_elapsed):
                    next_step_output = await self._atake_next_step(
                        name_to_tool_map,
                        color_mapping,
                        inputs,
                        intermediate_steps,
                        run_manager=run_manager,
                    )
                    if isinstance(next_step_output, AgentFinish):
                        return await self._areturn(
                            next_step_output,
                            intermediate_steps,
                            run_manager=run_manager,
                        )

                    intermediate_steps.extend(next_step_output)
                    if len(next_step_output) == 1:
                        next_step_action = next_step_output[0]
                        # See if tool should return directly
                        tool_return = self._get_tool_return(next_step_action)
                        if tool_return is not None:
                            return await self._areturn(
                                tool_return, intermediate_steps, run_manager=run_manager
                            )

                    iterations += 1
                    time_elapsed = time.time() - start_time
                output = self.agent.return_stopped_response(
                    self.early_stopping_method, intermediate_steps, **inputs
                )
                return await self._areturn(
                    output, intermediate_steps, run_manager=run_manager
                )
            except TimeoutError:
                # stop early when interrupted by the async timeout
                output = self.agent.return_stopped_response(
                    self.early_stopping_method, intermediate_steps, **inputs
                )
                return await self._areturn(
                    output, intermediate_steps, run_manager=run_manager
                )

    def _get_tool_return(
        self, next_step_output: Tuple[AgentAction, str]
    ) -> Optional[AgentFinish]:
        """Check if the tool is a returning tool."""
        agent_action, observation = next_step_output
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        # Invalid tools won't be in the map, so we return False.
        if agent_action.tool in name_to_tool_map:
            if name_to_tool_map[agent_action.tool].return_direct:
                return AgentFinish(
                    {self.agent.return_values[0]: observation},
                    "",
                )
        return None

    def _prepare_intermediate_steps(
        self, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> List[Tuple[AgentAction, str]]:
        # trim_intermediate_steps默认等于-1；也就是默认返回intermediate_steps
        if (
            isinstance(self.trim_intermediate_steps, int)
            and self.trim_intermediate_steps > 0
        ):
            return intermediate_steps[-self.trim_intermediate_steps :]
        elif callable(self.trim_intermediate_steps):
            return self.trim_intermediate_steps(intermediate_steps)
        else:
            return intermediate_steps