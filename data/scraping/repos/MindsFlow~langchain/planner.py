"""Agent that interacts with OpenAPI APIs via a hierarchical planning approach."""
import json
import re
from typing import List, Optional
from langchain.callbacks.stream_web import StreamingWebCallbackHandler

from functools import partial
from typing import Any, Callable, Dict, List, Optional

import yaml
from pydantic import Field

from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits.openapi.planner_prompt import (
    API_CONTROLLER_PROMPT,
    API_CONTROLLER_TOOL_DESCRIPTION,
    API_CONTROLLER_TOOL_NAME,
    API_ORCHESTRATOR_PROMPT,
    API_PLANNER_PROMPT,
    API_PLANNER_TOOL_DESCRIPTION,
    API_PLANNER_TOOL_NAME,
    PARSING_DELETE_PROMPT,
    PARSING_GET_PROMPT,
    PARSING_PATCH_PROMPT,
    PARSING_POST_PROMPT,
    REQUESTS_DELETE_TOOL_DESCRIPTION,
    REQUESTS_GET_TOOL_DESCRIPTION,
    REQUESTS_PATCH_TOOL_DESCRIPTION,
    REQUESTS_POST_TOOL_DESCRIPTION,
)
from langchain.agents.agent_toolkits.openapi.spec import ReducedOpenAPISpec
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.tools import Tool
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain
from langchain.llms.openai import OpenAI
from langchain.memory import ReadOnlySharedMemory
from langchain.prompts import PromptTemplate
from langchain.prompts.base import BasePromptTemplate
from langchain.requests import RequestsWrapper
from langchain.tools.base import BaseTool
from langchain.tools.requests.tool import BaseRequestsTool

#
# Requests tools with LLM-instructed extraction of truncated responses.
#
# Of course, truncating so bluntly may lose a lot of valuable
# information in the response.
# However, the goal for now is to have only a single inference step.
MAX_RESPONSE_LENGTH = 5000


def parse_text(text):
    text = text.strip("```json")
    text = text.strip("```")
    text = text.replace("```", "")
    text = text.replace("\n", "")
    text = text.replace("\t", "")
    text = text.replace(" ", "")
    return text


def _get_default_llm_chain(prompt: BasePromptTemplate) -> LLMChain:
    return LLMChain(
        llm=OpenAI(),
        prompt=prompt,
    )


def _get_default_llm_chain_factory(
        prompt: BasePromptTemplate,
) -> Callable[[], LLMChain]:
    """Returns a default LLMChain factory."""
    return partial(_get_default_llm_chain, prompt)


class RequestsGetToolWithParsing(BaseRequestsTool, BaseTool):
    name = "requests_get"
    description = REQUESTS_GET_TOOL_DESCRIPTION
    response_length: Optional[int] = MAX_RESPONSE_LENGTH
    llm_chain: LLMChain = Field(
        default_factory=_get_default_llm_chain_factory(PARSING_GET_PROMPT)
    )

    def _run(self, text: str) -> str:
        try:
            text = parse_text(text)
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise e
        data_params = data.get("params")
        response = self.requests_wrapper.get(data["url"], params=data_params)
        response = response[: self.response_length]
        return self.llm_chain.predict(
            response=response, instructions=data["output_instructions"]
        ).strip()

    async def _arun(self, text: str) -> str:
        return self._run(text)


class RequestsPostToolWithParsing(BaseRequestsTool, BaseTool):
    name = "requests_post"
    description = REQUESTS_POST_TOOL_DESCRIPTION

    response_length: Optional[int] = MAX_RESPONSE_LENGTH
    llm_chain: LLMChain = Field(
        default_factory=_get_default_llm_chain_factory(PARSING_POST_PROMPT)
    )

    def _run(self, text: str) -> str:
        try:
            text = parse_text(text)
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise e
        response = self.requests_wrapper.post(data["url"], data["data"])
        response = response[: self.response_length]
        return self.llm_chain.predict(
            response=response, instructions=data["output_instructions"]
        ).strip()

    async def _arun(self, text: str) -> str:
        return self._run(text)


class RequestsPatchToolWithParsing(BaseRequestsTool, BaseTool):
    name = "requests_patch"
    description = REQUESTS_PATCH_TOOL_DESCRIPTION

    response_length: Optional[int] = MAX_RESPONSE_LENGTH
    llm_chain: LLMChain = Field(
        default_factory=_get_default_llm_chain_factory(PARSING_PATCH_PROMPT)
    )

    def _run(self, text: str) -> str:
        try:
            text = parse_text(text)
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise e
        response = self.requests_wrapper.patch(data["url"], data["data"])
        response = response[: self.response_length]
        return self.llm_chain.predict(
            response=response, instructions=data["output_instructions"]
        ).strip()

    async def _arun(self, text: str) -> str:
        return self._run(text)


class RequestsDeleteToolWithParsing(BaseRequestsTool, BaseTool):
    name = "requests_delete"
    description = REQUESTS_DELETE_TOOL_DESCRIPTION

    response_length: Optional[int] = MAX_RESPONSE_LENGTH
    llm_chain: LLMChain = Field(
        default_factory=_get_default_llm_chain_factory(PARSING_DELETE_PROMPT)
    )

    def _run(self, text: str) -> str:
        try:
            text = parse_text(text)
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise e
        response = self.requests_wrapper.delete(data["url"])
        response = response[: self.response_length]
        return self.llm_chain.predict(
            response=response, instructions=data["output_instructions"]
        ).strip()

    async def _arun(self, text: str) -> str:
        return self._run(text)


#
# Orchestrator, planner, controller.
#
def _create_api_planner_tool(
        api_spec: ReducedOpenAPISpec, llm: BaseLanguageModel, plugin: Optional[dict]
) -> Tool:
    endpoint_descriptions = [
        f"{name} {description}" for name, description, _ in api_spec.endpoints
    ]
    prompt = PromptTemplate(
        template=API_PLANNER_PROMPT,
        input_variables=["query"],
        partial_variables={"endpoints": "- " + "- ".join(endpoint_descriptions)},
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    tool = Tool(
        name=plugin["name"] + " " + API_PLANNER_TOOL_NAME,
        description=API_PLANNER_TOOL_DESCRIPTION.format(plugin["name"], plugin["description"]),
        coroutine=chain.arun,
        func=chain.run
    )
    return tool


def _create_api_controller_agent(
        api_url: str,
        api_docs: str,
        requests_wrapper: RequestsWrapper,
        llm: BaseLanguageModel,
        headers: dict
) -> AgentExecutor:
    requests_wrapper.headers = headers
    get_llm_chain = LLMChain(llm=llm, prompt=PARSING_GET_PROMPT)
    post_llm_chain = LLMChain(llm=llm, prompt=PARSING_POST_PROMPT)
    tools: List[BaseTool] = [
        RequestsGetToolWithParsing(
            requests_wrapper=requests_wrapper, llm_chain=get_llm_chain
        ),
        RequestsPostToolWithParsing(
            requests_wrapper=requests_wrapper, llm_chain=post_llm_chain
        ),
    ]
    prompt = PromptTemplate(
        template=API_CONTROLLER_PROMPT,
        input_variables=["input", "agent_scratchpad"],
        partial_variables={
            "api_url": api_url,
            "api_docs": api_docs,
            "tool_names": ", ".join([tool.name for tool in tools]),
            "tool_descriptions": "\n".join(
                [f"{tool.name}: {tool.description}" for tool in tools]
            ),
        },
    )
    agent = ZeroShotAgent(
        llm_chain=LLMChain(llm=llm, prompt=prompt),
        allowed_tools=[tool.name for tool in tools],
    )
    return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)


def _create_api_controller_tool(
        api_spec: ReducedOpenAPISpec,
        requests_wrapper: RequestsWrapper,
        llm: BaseLanguageModel,
        plugin: Optional[dict]
) -> Tool:
    """Expose controller as a tool.

    The tool is invoked with a plan from the planner, and dynamically
    creates a controller agent with relevant documentation only to
    constrain the context.
    """

    base_url = api_spec.servers[0]["url"]  # TODO: do better.

    def check_name_matches_names(names: dict, name: str) -> tuple:
        for endpoint_key in names.keys():
            method, path = endpoint_key.split(" ", 1)
            name_method, name_path = name.split(" ", 1)

            if method != name_method:
                continue

            endpoint_parts = path.strip("/").split("/")
            name_parts = name_path.strip("/").split("/")

            if len(endpoint_parts) != len(name_parts):
                continue

            match = True

            for endpoint_part, name_part in zip(endpoint_parts, name_parts):
                if endpoint_part.startswith("{") and endpoint_part.endswith("}"):
                    param_info = names[endpoint_key]
                    param_name = endpoint_part[1:-1]

                    param = None
                    for p in param_info["parameters"]:
                        if p["name"] == param_name:
                            param = p

                    if param and "enum" in param.get("schema", {}):
                        if name_part not in param["schema"]["enum"]:
                            match = False
                            break
                elif endpoint_part != name_part:
                    match = False
                    break

            if match:
                return True, endpoint_key

        return False, None

    def _create_and_run_api_controller_agent(plan_str: str) -> str:
        pattern = r"\b(GET|POST|PATCH|DELETE)\s+(/\S+)*"
        matches = re.findall(pattern, plan_str)
        endpoint_names = [
            "{method} {route}".format(method=method, route=route.split("?")[0])
            for method, route in matches
        ]
        endpoint_docs_by_name = {name: docs for name, _, docs in api_spec.endpoints}
        docs_str = ""
        for endpoint_name in endpoint_names:
            matching, matched_key = check_name_matches_names(endpoint_docs_by_name, endpoint_name)
            if not matching:
                raise ValueError(f"{endpoint_name} endpoint does not exist.")

            docs_str += f"== Docs for {endpoint_name} == \n{yaml.dump(endpoint_docs_by_name.get(matched_key))}\n"
        agent = _create_api_controller_agent(base_url, docs_str, requests_wrapper, llm, plugin["headers"])
        return agent.run(plan_str)

    async def _acreate_and_run_api_controller_agent(plan_str: str) -> str:
        pattern = r"\b(GET|POST)\s+(/\S+)*"
        matches = re.findall(pattern, plan_str)
        endpoint_names = [
            "{method} {route}".format(method=method, route=route.split("?")[0])
            for method, route in matches
        ]
        endpoint_docs_by_name = {name: docs for name, _, docs in api_spec.endpoints}
        docs_str = ""
        for endpoint_name in endpoint_names:
            matching, matched_key = check_name_matches_names(endpoint_docs_by_name, endpoint_name)
            if not matching:
                raise ValueError(f"{endpoint_name} endpoint does not exist.")

            docs_str += f"== Docs for {endpoint_name} == \n{yaml.dump(endpoint_docs_by_name.get(matched_key))}\n"
        agent = _create_api_controller_agent(base_url, docs_str, requests_wrapper, llm, plugin["headers"])
        return await agent.arun(plan_str)

    return Tool(
        name=plugin["name"] + " " + API_CONTROLLER_TOOL_NAME,
        coroutine=_acreate_and_run_api_controller_agent,
        description=API_CONTROLLER_TOOL_DESCRIPTION.format(plugin["name"]),
        func=_create_and_run_api_controller_agent
    )


def create_openapi_agent(
    api_spec: ReducedOpenAPISpec,
    requests_wrapper: RequestsWrapper,
    llm: BaseLanguageModel,
    shared_memory: Optional[ReadOnlySharedMemory] = None,
    callback_manager: Optional[BaseCallbackManager] = None,
    verbose: bool = True,
    agent_executor_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Dict[str, Any],
) -> AgentExecutor:
    """Instantiate API planner and controller for a given spec.

    Inject credentials via requests_wrapper.

    We use a top-level "orchestrator" agent to invoke the planner and controller,
    rather than a top-level planner
    that invokes a controller with its plan. This is to keep the planner simple.
    """
    tools = [
        _create_api_planner_tool(api_spec, llm),
        _create_api_controller_tool(api_spec, requests_wrapper, llm),
    ]
    prompt = PromptTemplate(
        template=API_ORCHESTRATOR_PROMPT,
        input_variables=["input", "agent_scratchpad"],
        partial_variables={
            "tool_names": ", ".join([tool.name for tool in tools]),
            "tool_descriptions": "\n".join(
                [f"{tool.name}: {tool.description}" for tool in tools]
            ),
        },
    )
    agent = ZeroShotAgent(
        llm_chain=LLMChain(llm=llm, prompt=prompt, memory=shared_memory),
        allowed_tools=[tool.name for tool in tools],
        **kwargs,
    )
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        callback_manager=callback_manager,
        verbose=verbose,
        **(agent_executor_kwargs or {}),
    )


def create_openapi_custom_agent(
        api_spec: ReducedOpenAPISpec,
        requests_wrapper: RequestsWrapper,
        llm: BaseLanguageModel,
        plugin: dict,
        custom_tool: BaseTool,
        memory: Optional[ReadOnlySharedMemory] = None,
        callback_manager: Optional[BaseCallbackManager] = None,
        verbose: bool = True,
        agent_executor_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Dict[str, Any],
) -> AgentExecutor:
    tools = []
    if api_spec is not None:
        tools.append(_create_api_planner_tool(api_spec, llm, plugin))
        tools.append(_create_api_controller_tool(api_spec, requests_wrapper, llm, plugin))
    # tools.append(get_gpt_tool(llm))
    if custom_tool is not None:
        tools.append(custom_tool)
    prompt = PromptTemplate(
        template=API_ORCHESTRATOR_PROMPT,
        input_variables=["input", "agent_scratchpad"],
        partial_variables={
            "tool_names": ", ".join([tool.name for tool in tools]),
            "tool_descriptions": "\n".join(
                [f"{tool.name}: {tool.description}" for tool in tools]
            ),
        },
    )
    agent = ZeroShotAgent(
        llm_chain=LLMChain(llm=llm, prompt=prompt, memory=memory),
        allowed_tools=[tool.name for tool in tools],
        **kwargs,
    )
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        callback_manager=callback_manager,
        verbose=verbose,
        **(agent_executor_kwargs or {}),
    )


def get_gpt_tool(llm):
    prompt = PromptTemplate(
        template="if all other tools can not answer user's question, this tool can help user to answer question\n{query}",
        input_variables=["query"],
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    tool = Tool(
        name="AnyGPT",
        description="if all other tools can not answer user's question, gpt will help user to answer question",
        coroutine=chain.arun,
        func=chain.run
    )
    return tool
