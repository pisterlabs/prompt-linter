from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Type

from pydantic import Field, BaseModel 

from langchain.tools.base import BaseTool
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.callbacks.stdout import StdOutCallbackHandler
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.chat_models.openai import ChatOpenAI

from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from bespokebots.services.chains.prompts import CalendarAnalysisChatPrompt
from bespokebots.services.chains import (
    CalendarAnalysisLlmChain
)

from bespokebots.services.chains.calendar_templates import (
    CalendarDataAnalyzerTemplates as Templates
)

from bespokebots.services.chains.output_parsers import (
    CalendarAnalyzerOutputParserFactory as OutputParserFactory
)


class CalendarDataAnalyzerSchema(BaseModel):
    """Schema for the CalendarDataAnalyzerTool."""
    events: str = Field(
        ..., 
        title="Events",
        description="The calendar event data to analyze, in JSON format."
    )
    user_question: str = Field(
        ..., 
        title="User Question",
        description="The question about their calendar the user is trying to get an answer too."
    )

    user_requirements: Optional[str] = Field(
        None, 
        title="User Requirements",
        description="Additional requirements the user wants the assistant to acccount for."
    )

class CalendarDataAnalyzerTool(BaseTool):
    """Tool wrapped around an LLM primed for calendar data analysis via the CalendarDataAnalyzerChain factory."""
    name: str = "calendar_data_analyzer"
    description: str = """Tool intended to be used to answer questions about a user's calendar data based on a set of events
    provided to the tool. This tool can help figure out answers to scheduling questions.  It can also help count up days, weeks
    or months over a given time period."""
    
    args_schema: Type[CalendarDataAnalyzerSchema] = CalendarDataAnalyzerSchema
    llm: ChatOpenAI = ChatOpenAI(temperature=0, model_name="gpt-4") #this is the magic right here
    prompt: ChatPromptTemplate = None
    output_parser = OutputParserFactory.create_output_parser()

    def _run(
        self, 
        events: str, 
        user_question: str, 
        user_requirements: Optional[str] = None
    ) -> str:
        """Run the tool."""
        if self.prompt is None:
            self.prompt = CalendarAnalysisChatPrompt.from_user_request(user_question)
        

        chain = CalendarAnalysisLlmChain.build_calendar_chain(
                    prompt = self.prompt,
                    llm = self.llm,
                    output_parser = self.output_parser
                )
        chain.verbose = True
        output = chain.run_chain(events, CalendarDataAnalyzerTool.default_additional_user_requirements()) 

        #  # print("---------------------------------------------------------------------------------")
        return output
    
    async def _arun(self, calendar_id: str, summary: str, event_id: str, start_time: str, end_time: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> dict:
        raise NotImplementedError(f"The tool {self.name} does not support async yet.")
    
    @staticmethod
    def default_additional_user_requirements() -> str:
        """Return default additional user requirements."""
        user_requirements = [
        "- Ensure there is no less than 30 minutes of free time before and after any event that needs to be scheduled.",
        "- On any given day, there should be as few appointments as possible, so when choosing between two or more days on which to schedule an appointment, choose the day with the least number of appointments.",
        ]
        separator = "\n\t"
        return Templates.USER_REQUIREMENTS_TEMPLATE(
            separator.join(user_requirements)   
        )    