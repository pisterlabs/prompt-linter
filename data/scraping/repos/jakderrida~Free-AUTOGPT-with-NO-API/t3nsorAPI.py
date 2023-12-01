from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
import t3nsor

class gpt3NoInternet(LLM):
    messages: Optional[List[Mapping[str, Any]]] = []
    
    @property
    def _llm_type(self) -> str:
        return "custom"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        
        t3nsor_cmpl = t3nsor.Completion.create(
            prompt   = prompt,
            messages = self.messages
        )

        response = t3nsor_cmpl.completion.choices[0].text
        
        self.messages.append({'role': 'user', 'content': prompt})
        self.messages.append({'role': 'assistant', 'content': response})
        
        return response
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model": "GPT3.5Turbo"}

#llm = gpt3NoInternet()

#print(llm("Hello, how are you?"))
#print(llm("what is AI?"))
#print(llm("how have i question in before?"))
