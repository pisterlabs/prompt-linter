import os
import google.generativeai as palm
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun

palm.configure(api_key=os.getenv('PALM_API_KEY'))

class PALM(LLM):

    @property
    def _llm_type(self) -> str:
        return "PALM API"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        
        models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
        model = models[0].name

        completion = palm.generate_text(
            model=model,
            prompt=prompt,
            temperature=0,
            # The maximum length of the response
            max_output_tokens=800,
        )

        return completion.result
