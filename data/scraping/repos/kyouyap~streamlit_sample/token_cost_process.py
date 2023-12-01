from langchain.callbacks.base import AsyncCallbackHandler
from langchain.schema import LLMResult
from typing import Any, Dict, List
import tiktoken

MODEL_COST_PER_1K_TOKENS = {
    "gpt-4": 0.03,
    "gpt-4-0314": 0.03,
    "gpt-4-completion": 0.06,
    "gpt-4-0314-completion": 0.06,
    "gpt-4-32k": 0.06,
    "gpt-4-32k-0314": 0.06,
    "gpt-4-32k-completion": 0.12,
    "gpt-4-32k-0314-completion": 0.12,
    "gpt-3.5-turbo": 0.002,
    "gpt-3.5-turbo-0301": 0.002,
    "text-ada-001": 0.0004,
    "ada": 0.0004,
    "text-babbage-001": 0.0005,
    "babbage": 0.0005,
    "text-curie-001": 0.002,
    "curie": 0.002,
    "text-davinci-003": 0.02,
    "text-davinci-002": 0.02,
    "code-davinci-002": 0.02,
}

class TokenCostProcess:
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    successful_requests: int = 0

    def sum_prompt_tokens( self, tokens: int ):
      self.prompt_tokens = self.prompt_tokens + tokens
      self.total_tokens = self.total_tokens + tokens

    def sum_completion_tokens( self, tokens: int ):
      self.completion_tokens = self.completion_tokens + tokens
      self.total_tokens = self.total_tokens + tokens

    def sum_successful_requests( self, requests: int ):
      self.successful_requests = self.successful_requests + requests

    def get_openai_total_cost_for_model( self, model: str ) -> float:
       return MODEL_COST_PER_1K_TOKENS[model] * self.total_tokens / 1000
    
    def get_cost_summary(self, model:str) -> str:
        cost = self.get_openai_total_cost_for_model(model)

        return cost

class CostCalcAsyncHandler(AsyncCallbackHandler):
    model: str = ""
    socketprint = None
    websocketaction: str = "appendtext"
    token_cost_process: TokenCostProcess

    def __init__( self, model, token_cost_process ):
       self.model = model
       self.token_cost_process = token_cost_process

    def on_llm_start( self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
       encoding = tiktoken.encoding_for_model( self.model )

       if self.token_cost_process == None: return

       for prompt in prompts:
          self.token_cost_process.sum_prompt_tokens( len(encoding.encode(prompt)) )

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
      print( token )

      self.token_cost_process.sum_completion_tokens( 1 )

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
      self.token_cost_process.sum_successful_requests( 1 )
     