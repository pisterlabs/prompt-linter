import sagemaker
import ai21
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any, Dict

# need to deploy jurassic jumbo instruct model to sagemaker endpoint first

class SageMakerLLM(LLM):
    
    @property
    def _llm_type(self) -> str:
        return "jurassic-jumbo-instruct"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = ai21.Completion.execute(
            sm_endpoint="j2-jumbo-instruct",
            prompt=prompt,
            maxTokens=500,
            temperature=0,
            numResults=1,
            stopSequences=stop,
        )
        return response['completions'][0]['data']['text']