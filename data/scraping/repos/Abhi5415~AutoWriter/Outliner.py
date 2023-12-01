from __future__ import annotations
from typing import List, Optional, Union
from pydantic import ValidationError
from langchain.chains.llm import LLMChain
from langchain.chat_models.base import BaseChatModel
from langchain.experimental.autonomous_agents.autogpt.output_parser import (
    AutoGPTOutputParser,
    BaseAutoGPTOutputParser,
)
from langchain.experimental.autonomous_agents.autogpt.prompt import AutoGPTPrompt
from langchain.experimental.autonomous_agents.autogpt.prompt_generator import (
    FINISH_NAME,
)
from langchain.schema import (
    AIMessage,
    BaseMessage,
    Document,
    HumanMessage,
    SystemMessage,
)
from langchain.tools.base import BaseTool
from langchain.tools.human.tool import HumanInputRun
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from utils.StageReturnType import StageReturnType, Stage
from BaseContent import BaseContent, FeedbackInput, OutlineInput
from prompts.OutlinerPrompt import OutlinerPrompt
from langchain.utilities import SerpAPIWrapper



from langchain.agents import Tool

RESEARCH_TOOL_NAME = "Research"


class Outliner:
    """Agent class for interacting with Auto-GPT."""

    def __init__(
        self,
        memory: VectorStoreRetriever,
        chain: LLMChain,
        output_parser: BaseAutoGPTOutputParser,
        tools: List[BaseTool],
        feedback_tool: Optional[HumanInputRun] = None,
    ):
        self.memory = memory
        self.full_message_history: List[BaseMessage] = []
        self.next_action_count = 0
        self.chain = chain
        self.output_parser = output_parser
        self.tools = tools
        self.feedback_tool = feedback_tool

    @classmethod
    def from_llm_and_tools(
        cls,
        memory: VectorStoreRetriever,
        content: BaseContent,
        llm: BaseChatModel,
        output_parser: Optional[BaseAutoGPTOutputParser] = None,
    ) -> Outliner:
        
        search = SerpAPIWrapper()
        tools = [
            Tool(
                name = "search",
                func=search.run,
                description="useful for when you need to answer questions about current events. You should ask targeted questions"
            ),
            HumanInputRun(),
            Tool(
                name=RESEARCH_TOOL_NAME,
                description="request further research on a topic if needed",
                func=lambda x: x,
                args_schema=FeedbackInput
            ),
            Tool(
                name="write_outline",
                description="use this command to save your outline to a file",
                func=content.writeOutline,
                args_schema=OutlineInput
            )
        ]

        prompt = OutlinerPrompt(
            tools=tools,
            input_variables=["content", "feedback", "messages", "memory", "user_input"],
            token_counter=llm.get_num_tokens,
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        return cls(
            memory,
            chain,
            output_parser or AutoGPTOutputParser(),
            tools,
        )

    def run(self, content: BaseContent, feedback: Union[str, None]) -> StageReturnType:
        print("Running Outliner")
        user_input = (
            "Determine which next command to use, "
            "and respond using the format specified above:"
        )

        # Interaction Loop
        loop_count = 0
        while True:
            print(str(content))
            content.saveToFile()

            # Discontinue if continuous limit is reached
            loop_count += 1
            
            # Send message to AI, get response
            assistant_reply = self.chain.run(
                content=content,
                feedback=feedback,
                messages=self.full_message_history,
                memory=self.memory,
                user_input=user_input,
            )

            # Print Assistant thoughts
            print(assistant_reply)
            self.full_message_history.append(HumanMessage(content=user_input))
            self.full_message_history.append(AIMessage(content=assistant_reply))

            # Get command name and arguments
            action = self.output_parser.parse(assistant_reply)
            tools = {t.name: t for t in self.tools}
            if action.name == FINISH_NAME:
                return StageReturnType(
                    content=content,
                    feedback=None,
                    stage=Stage.WRITE
                )

            if action.name == RESEARCH_TOOL_NAME:
                return StageReturnType(
                    content=content,
                    feedback=None,
                    stage=Stage.RESEARCH
                )

            if action.name in tools:
                tool = tools[action.name]
                try:
                    observation = tool.run(action.args)
                except ValidationError as e:
                    observation = (
                        f"Validation Error in args: {str(e)}, args: {action.args}"
                    )
                except Exception as e:
                    observation = (
                        f"Error: {str(e)}, {type(e).__name__}, args: {action.args}"
                    )
                result = f"Command {tool.name} returned: {observation}"
            elif action.name == "ERROR":
                result = f"Error: {action.args}. "
            else:
                result = (
                    f"Unknown command '{action.name}'. "
                    f"Please refer to the 'COMMANDS' list for available "
                    f"commands and only respond in the specified JSON format."
                )

            memory_to_add = (
                f"Assistant Reply: {assistant_reply} " f"\nResult: {result} "
            )
            self.memory.add_documents([Document(page_content=memory_to_add)])
            self.full_message_history.append(SystemMessage(content=result))

