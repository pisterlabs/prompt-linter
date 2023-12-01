from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Extra
import json
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain import LLMChain
from langchain.output_parsers import PydanticOutputParser

from bespokebots.services.chains.prompts.calendar_analysis_chat_prompt import (
    CalendarAnalysisChatPrompt,
)
from bespokebots.services.chains.calendar_templates import (
    CalendarDataAnalyzerTemplates as Templates,
)
from bespokebots.services.chains.output_parsers import (
    CalendarAnalyzerOutputParserFactory,
)


class CalendarAnalysisLlmChain:
    """Wrapper around an LLM Chain which will be used to analyze a user's calendar data,
    hopefully making it easier and quicker for an LLM to answer user's questions
    regarding their schedule and when it makes sense to schedule events for."""

    def __init__(self, prompt, llm, output_parser=None):
        self.prompt: CalendarAnalysisChatPrompt = prompt
        """Prompt object to use."""
        self.llm: BaseLanguageModel = llm
        """Language model to use."""
        self.output_parser: PydanticOutputParser = (
            CalendarAnalyzerOutputParserFactory.create_output_parser()
            if output_parser is None
            else output_parser
        )

    @staticmethod
    def build_calendar_chain(
        prompt, llm, output_parser=None
    ) -> CalendarAnalysisLlmChain:
        """Build a CalendarAnalysisLlmChain object."""
        return CalendarAnalysisLlmChain(
            prompt=prompt,
            llm=llm,
        )

    def run_chain(self, events: str, user_requirements: str):
        """Wraps around LLMChain's run semantics to ensure the OutputParser is used."""
        

        chain = LLMChain(llm=self.llm, prompt=self.prompt)
        output = chain(
            inputs={
                "events_to_analyze": events,
                "user_requirements": user_requirements,
                "output_parser_template": self.output_parser.get_format_instructions(),
            }
        )
        print("-----------------CHAIN OUTPUT-----------------")
        [
            print(f"\n----------------\n{key}==> <[[ {value} ]]\n----------------\n")
            for key, value in output.items()
        ]
        print("-----------------END CHAIN OUTPUT-----------------")

        #the output isn't just JSON, its a big blob of text.  Fortunately, the JSON
        #chunks are delimited with three backticks: ``` so we can split on that
        
        #The outputchunks arrary should be indexed as follows:
        #0: The first chunk is a restatement of the original question, it ends with "Calendar Events:"
        #1: The second chunk is the JSON of the events to analyze
        #2: The third chunk is the LLM's list of what it did for each of the actions, which is interesting
        #3: The fourth chunk is the JSON response we need to parse with the output parser
        output_model = self.output_parser.parse(output["text"])
        return output_model.json()

    def find_json_answer(self, output_chunks: List[str]) -> str:
        """Find the JSON answer in the output chunks."""
        for chunk in output_chunks:
            thoughts_index = chunk.find('"thoughts":')
            if thoughts_index >= 0:
                first_bracket_index = chunk.find("{")
                last_bracket_index = chunk.rfind("}")
                try:
                    return json.loads(chunk[first_bracket_index:last_bracket_index])
                except json.JSONDecodeError:
                    pass
        raise ValueError("No JSON found in output chunks.")
