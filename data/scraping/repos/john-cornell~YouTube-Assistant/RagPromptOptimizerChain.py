from typing import List
from dotenv import load_dotenv
from multiagent.rag_prompt_agent import rag_prompt_agent, rag_prompt_agent_input
from multiagent.json_extractor import extract_json, get_error_json
from typing import Any, Dict, List, Optional
import json

from langchain.schema.language_model import BaseLanguageModel
from langchain.chains.base import Chain
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun
)

load_dotenv()

class RagPromptOptimizerChain(Chain):
    llm:BaseLanguageModel
    debug: bool

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"

    @property
    def input_keys(self) -> List[str]:
        return ["query"]

    @property
    def output_keys(self) -> List[str]:
        return ["response"]

    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None
    ) -> Dict[str, str]:

        prompt_to_analyse = inputs["query"]

        if not prompt_to_analyse or prompt_to_analyse.strip() == '':
            raise ValueError("prompt_to_analyse not set")

        agent = rag_prompt_agent(llm=self.llm)

        try:
            output = agent.run(
                input=rag_prompt_agent_input(prompt_to_analyse)
            )
        except Exception as e:
            print(f"Agent error: {e}")
            return {"response": get_error_json(f"Agent error: {e}", output_key="searchprompt")}

        output = extract_json(output, output_key="searchprompt")

        output_dict = json.loads(output)

        thinking = output_dict["thinking"]
        process = output_dict["searchprompt"]

        if self.debug:
            print(f"Thinking: {thinking}")
            print()
            print(f"searchprompt: {process}")

        return {"response": json.loads(output)}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:

        prompt_to_analyse = inputs["query"]

        if not prompt_to_analyse or prompt_to_analyse.strip() == '':
            raise ValueError("prompt_to_analyse not set")

        agent = rag_prompt_agent(llm=self.llm)

        try:
            output = agent.arun(
                input=rag_prompt_agent_input(prompt_to_analyse)
            )
        except Exception as e:
            print(f"Agent error: {e}")
            return {"response": get_error_json(f"Agent error: {e}", output_key="searchprompt")}

        output = extract_json(output, output_key="searchprompt")

        output_dict = json.loads(output)

        thinking = output_dict["thinking"]
        process = output_dict["searchprompt"]

        if self.debug:
            print(f"Thinking: {thinking}")
            print()
            print(f"Search Prompt: {process}")

        return {"response": json.loads(output)}

    @property
    def _chain_type(self) -> str:
        return "Rag Prompt Optimization Chain"


