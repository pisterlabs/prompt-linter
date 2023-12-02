from typing import Dict

from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts import HumanMessagePromptTemplate, MessagesPlaceholder, ChatPromptTemplate
from langchain.schema import SystemMessage
from langchain.tools import BaseTool
from yid_langchain_extensions.agent.simple_agent import SimpleAgent
from yid_langchain_extensions.output_parser.action_parser import ActionParser
from yid_langchain_extensions.tools.agent_as_tool import AgentAsTool
from yid_langchain_extensions.tools.utils import FinalAnswerTool, format_tool_names, format_tools

from agents.tools import WebSearchTool, AskPagesTool
from agents.utils import format_now, get_thought_thought, get_self_criticism_thought


class WebResearcherAgent:
    def __init__(self, prompts: Dict[str, str]):
        self.prompts = prompts
        self.smart_llm = ChatOpenAI(model_name="gpt-4-0613", temperature=0)
        self.fast_llm = ChatOpenAI(model_name="gpt-3.5-turbo-0613", temperature=0)
        final_answer_tool = FinalAnswerTool()
        web_search_tool = WebSearchTool()
        ask_url_tool = AskPagesTool(llm=self.smart_llm)
        self.tools = [final_answer_tool, web_search_tool, ask_url_tool]
        self.output_parser = ActionParser.from_extra_thoughts(pre_thoughts=[
            get_thought_thought(), get_self_criticism_thought()
        ], after_thoughts=[])
        self.format_message = PromptTemplate.from_template(
            self.output_parser.get_format_instructions(), template_format="jinja2").format(
            tool_names=format_tool_names(self.tools)
        )

    def as_tool(self) -> BaseTool:
        name: str = "web_search"
        description: str = self.prompts["as_tool_intro"]

        system_message = PromptTemplate.from_template(self.prompts["prefix"], template_format="jinja2").format(
            date=format_now()
        )
        messages = [
            SystemMessage(content=system_message),
            HumanMessagePromptTemplate.from_template("{{input}}", "jinja2"),
        ]
        messages.extend([
            MessagesPlaceholder(variable_name="agent_scratchpad"),
            SystemMessage(content=format_tools(self.tools)),
            SystemMessage(content=self.format_message),
        ])
        prompt = ChatPromptTemplate.from_messages(messages=messages)
        agent_executor = SimpleAgent.from_llm_and_prompt(
            llm=self.smart_llm,
            prompt=prompt,
            output_parser=self.output_parser,
            stop_sequences=self.output_parser.stop_sequences,
        ).get_executor(tools=self.tools, verbose=True)
        return AgentAsTool(
            name=name,
            description=description,
            return_direct=False,
            executor=agent_executor,
            adapter=lambda *args, **kwargs: ((), {"input": args[0]}),
        )
