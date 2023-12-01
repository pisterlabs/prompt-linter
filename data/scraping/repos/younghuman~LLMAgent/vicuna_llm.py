from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatGeneration,
    ChatResult,
)
from langchain.chat_models.base import BaseChatModel
from typing import Optional, List, Mapping, Any
from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
import replicate

class VicunaLLM(BaseChatModel):      
    model_id: str = "replicate/vicuna-13b:e6d469c2b11008bb0e446c3e9629232f9674581224536851272c54871f84076e"
    repetition_penalty: int = 5
    temperature: float = 0.01
    top_p: int = 1
    max_length: int = 1024
    
    @property
    def _llm_type(self) -> str:
        return "vicuna"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> ChatResult:
        prompt = "\n".join(m.content for m in messages)
        
        input_dict = {
            "prompt": prompt,
            "repetition_penalty": self.repetition_penalty,
            "temperature": self.temperature,
            
        }
        response = replicate.run(
            self.model_id,
            input=input_dict
        )
        content = "".join(response)
        message = AIMessage(content=content)
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    ) -> ChatResult:
        prompt = "\n".join(m.content for m in messages)
        
        input_dict = {"prompt": prompt}
        input_dict.update(self.llm_input)
        response = await replicate.run(
            self.model_id,
            input=input_dict
        )
        content = "".join(response)
        message = AIMessage(content=content)
        return ChatResult(generations=[ChatGeneration(message=message)])
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_id": self.model_id
        }