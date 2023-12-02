from app.task_resolver.engine import StepResolver

from typing import List, Any
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

# from langchain.chat_models import ChatOpenAI
from app.utils import chain_verbose, logger
from langchain.llms import OpenAI

from app.tools import NextStepExtractor
from app.model import Message

from langchain.output_parsers import StructuredOutputParser, ResponseSchema

template="""Given a conversation between a user and an assistant about booking a house for short-term stay. \
Your job is to decide which is the next step to take.

Here are the steps for you to choose from:
{steps}

Current conversation: 
{chat_history}

{format_instructions}"""

response_schemas = [
    ResponseSchema(name="step", description="The name of the next step to take.")
]

class PostProcessRouterChain:

    def __init__(self):

        llm = OpenAI(temperature=0.)
        
        self.output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions =self.output_parser.get_format_instructions()

        prompt_template = PromptTemplate(
            input_variables=["chat_history", "steps"], 
            partial_variables={"format_instructions": format_instructions},
            template=template
        )

        self.chain = LLMChain(llm=llm, 
                              prompt=prompt_template, 
                              verbose=chain_verbose,
                              output_key="result")

    def run(self, chat_history: str, steps: str):
        info = self.chain({"chat_history": chat_history, "steps": steps})
        return self.output_parser.parse(info["result"])
    


class PostProcessRouterResolver(StepResolver):

    def __init__(self, steps):
        self.steps = steps
        self.next_step_extractor = NextStepExtractor()

    def run(self, messages: List[Message], previous_steps_data: dict=None, step_chat_history: List[Message] = None) -> Message:
        
        result = self.next_step_extractor.run_select_next_step(messages, self.steps)
        return Message.route_message("Routing to previous Step", result["step_id"]) 
        
    def is_done(self):
        return True
            