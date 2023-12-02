from __future__ import annotations
import json
from typing import Any, Optional, Union
from langchain.prompts.chat import ChatPromptValue
from langchain.llms.base import LLM, BaseLLM
from langchain.chat_models.base import BaseChatModel
from langchain.embeddings.base import Embeddings
from langchain.schema import AIMessage, BaseMessage, ChatGeneration, ChatResult
from langchain.schema import PromptValue
from langchain.schema import LLMResult
from ..data_model import LlmPrompt, NamedPromptTemplateDescription
from ..decorators import FORMATTED_PROMPT_CONTEXT_KEY
from typing import List
from .. import PromptWatch
from langchain.schema import Generation
from typing import Any, Optional, Union
import datetime
from langchain.llms.base import LLM
from langchain.chat_models.base import BaseChatModel
from langchain.embeddings.base import Embeddings
from langchain.schema import HumanMessage, ChatMessage as LangChainChatMessage, AIMessage, SystemMessage, BaseMessage
from ..client import Client


from langchain.cache import BaseCache



class CachedLLM(LLM):
    """Cached LLM wrapper around the actual LLM."""
    inner_llm:Any
    cache_namespace_key:Optional[str]
    cache_embeddings:Optional[Embeddings]
    token_limit:Optional[int]
    similarity_limit:Optional[float]

       
    def __init__(self, inner_llm:BaseLLM, cache_namespace_key:str=None, cache_embeddings:Embeddings = None, token_limit:int=None, similarity_limit:float=0.97) -> None:
        super().__init__(inner_llm=inner_llm, cache_namespace_key=cache_namespace_key, cache_embeddings=cache_embeddings, token_limit=token_limit, similarity_limit=similarity_limit)
       
    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "cached-llm"
    
    
    
    def _get_from_cache(self,prompt, stop: Optional[List[str]] = None):
        promptwatch_context = PromptWatch.get_active_instance()

        if promptwatch_context:

            if isinstance(promptwatch_context.current_activity,LlmPrompt) and isinstance(promptwatch_context.current_activity.prompt_template, NamedPromptTemplateDescription):
                prompt_input_values = promptwatch_context.current_activity.prompt_input_values
                named_prompt_template = promptwatch_context.current_activity.prompt_template
            else:
                prompt_input_values=None
                named_prompt_template=None

            embed_func = self.cache_embeddings.embed_query if self.cache_embeddings else None
            cache = promptwatch_context.caching.get_or_init_cache(self.cache_namespace_key, embed_func, self.token_limit, self.similarity_limit)
            
            cache_prompt_req = f"Stop:[{','.join(stop)}]\n:{prompt}" if stop else prompt
            cached_res = cache.get(cache_prompt_req, prompt_template=named_prompt_template, prompt_input_values=prompt_input_values)
           
           
            return  cached_res, lambda cached_res,result : cache.add(cached_res, result) if cached_res is not None else None
        else:
            return None, None
        
    def _call(self, prompt:str, stop: Optional[List[str]] = None) -> str:
        """ Implementing abstract call method."""
        return self.generate([prompt], stop=stop).generations[0][0].text
    
      
        
    def _generate(
        self, prompts: List[str], stop: Optional[List[str]] = None
    ) -> LLMResult:
        if prompts and len(prompts)==1:
            # we do not support multiple prompts for now

            cached_res,callback=self._get_from_cache(prompts[0], stop=stop)
        
            if not cached_res:

                llmresult:ChatResult = self.inner_llm._generate(prompts, stop) 
                if callback:
                    callback(cached_res, llmresult.generations[0][0].text)
                return llmresult
            else:
                generation = Generation(text=cached_res.result, generation_info={"cached":True, **cached_res.metadata,  "cache_namespace_key":cached_res.cache_namespace_key})
                return LLMResult(generations=[[generation]], llm_output={"cached":True})
            
        else:
            return self.inner_llm._generate(prompts, stop) 
        
    async def _agenerate(self, 
                         prompts: List[str], stop: Optional[List[str]] = None
        ) -> LLMResult:
        if prompts and len(prompts)==1:
            cached_res,callback=self._get_from_cache(prompts[0], stop=stop)
            
            if not cached_res:

                chat_result:ChatResult = await self.inner_llm._agenerate(prompts, stop) 
                if callback:
                    callback(cached_res, chat_result.generations[0].text)
                return chat_result
            else:
                generation = Generation(text=cached_res.result, generation_info={"cached":True, **cached_res.metadata,  "cache_namespace_key":cached_res.cache_namespace_key})
                return LLMResult(generations=[[generation]], llm_output={"cached":True})
        else:
            return self.inner_llm._generate(prompts, stop) 


class CachedChatLLM(BaseChatModel):
    
    inner_llm:Any
    cache_namespace_key:Optional[str]
    cache_embeddings:Optional[Embeddings]
    token_limit:Optional[int]
    similarity_limit:Optional[float]
    def __init__(self, inner_llm:BaseLLM, cache_namespace_key:str=None, cache_embeddings:Embeddings = None, token_limit:int=None, similarity_limit:float=0.95) -> None:
        
        super().__init__(inner_llm=inner_llm, cache_namespace_key=cache_namespace_key, cache_embeddings=cache_embeddings, token_limit=token_limit, similarity_limit=similarity_limit)
       
    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "cached-chat-llm"
    
    def generate_prompt(
        self, prompts: List[PromptValue], stop: Optional[List[str]] = None, **kwargs
    ) -> LLMResult:
        # overriding generate_prompt because we want to pass down the prompts to the inner llm
        promptwatch_context = PromptWatch.get_active_instance()
        if promptwatch_context and len(prompts)==1 :
            promptwatch_context.add_context(FORMATTED_PROMPT_CONTEXT_KEY, prompts[0]) # .to_messages() or not .to_messages() ... that is the question
        return super().generate_prompt(prompts, stop=stop)
        
    
    def _get_from_cache(self, messages: Union[str, List[BaseMessage]], stop: Optional[List[str]] = None):
        promptwatch_context = PromptWatch.get_active_instance()
        prompt = ChatPromptValue(messages=messages).to_string()
        
        if promptwatch_context:
            
            embed_func = self.cache_embeddings.embed_query if self.cache_embeddings else None
            cache = promptwatch_context.caching.get_or_init_cache(self.cache_namespace_key, embed_func, self.token_limit, self.similarity_limit)
            
            
            if isinstance(promptwatch_context.current_activity,LlmPrompt) \
                and isinstance(promptwatch_context.current_activity.prompt_template, NamedPromptTemplateDescription) \
                and promptwatch_context.current_activity.prompt_template.template_name:
                    
                named_prompt_template = promptwatch_context.current_activity.prompt_template
                prompt_input_values = promptwatch_context.current_activity.prompt_input_values
            else:
                prompt_input_values=None
                named_prompt_template=None


            cache_prompt_req = f"Stop:[{','.join(stop)}]\n:{prompt}" if stop else prompt
            cached_res = cache.get(cache_prompt_req,prompt_template=named_prompt_template, prompt_input_values=prompt_input_values )
           
           
            return  cached_res, lambda cached_res,result : cache.add(cached_res, result) 
        else:
            return None, None
        
    def _call(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> str:
        """ Implementing abstract call method."""
        return self._generate(messages, stop=stop).generations[0].message
    
        
        
    async def _acall(self, messages: List[BaseMessage], stop: Optional[List[str]] = None) -> str:
        """ Implementing abstract call method."""
        return (await self._agenerate(messages, stop=stop)).generations[0].message
        
    async def _agenerate(
       self, messages: List[BaseMessage], stop: Optional[List[str]] = None,**kwargs
    ):
        cached_res,callback=self._get_from_cache(messages, stop=stop)
        
        if not cached_res:

            chat_result:ChatResult = await self.inner_llm._agenerate(messages, stop, **kwargs) 
            if callback:
                function_call = chat_result.generations[0].message and chat_result.generations[0].message.additional_kwargs and chat_result.generations[0].message.additional_kwargs.get("function_call")
                if function_call:
                    additional_data={"function_call":function_call}
                    result =  chat_result.generations[0].message.content + '\x03'+ json.dumps(additional_data)
                else:
                    result = chat_result.generations[0].message.content

                callback(cached_res, result)
            return chat_result
        else:
            if '\x03' in cached_res.result:
                result, additional_data = cached_res.result.split('\x03')
                additional_data = json.loads(additional_data)
                generated_msg = AIMessage(content=result, additional_kwargs=additional_data)
            else:
                generated_msg = AIMessage(content=cached_res.result)
            generation = ChatGeneration(message=generated_msg, generation_info={"cached":True, **cached_res.metadata, "cache_namespace_key":cached_res.cache_namespace_key})
            return ChatResult(generations=[generation],llm_output={"cached":True})

        
    def _generate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None,**kwargs
    ) -> ChatResult:
        cached_res,callback=self._get_from_cache(messages, stop=stop)
        
        
        if not cached_res:


            chat_result:ChatResult = self.inner_llm._generate(messages, stop,**kwargs) 
            if callback:
                callback(cached_res, chat_result.generations[0].text)
            return chat_result
        else:
            generated_msg = AIMessage(content=cached_res.result)
            generation = ChatGeneration(message=generated_msg, generation_info={"cached":True, **cached_res.metadata, "cache_namespace_key":cached_res.cache_namespace_key})
            return ChatResult(generations=[generation],llm_output={"cached":True})
    
    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        """Combine llm outputs."""
        if llm_outputs and len(llm_outputs)==1 and llm_outputs[0].get("cached"):
            return llm_outputs[0]
        else:
            return self.inner_llm._combine_llm_outputs(llm_outputs)



class PromptWatchLlmCache(BaseCache):

    def __init__(self, cache_namespace_key:str=None, cache_embeddings:Embeddings = None, token_limit:int=None, similarity_limit:float=0.97) -> None:
        
        self.cache_namespace_key = cache_namespace_key
        self.cache_embeddings = cache_embeddings
        self.similarity_limit=similarity_limit
        
        self.embed_func = self.cache_embeddings.embed_query if self.cache_embeddings else None

        self.token_limit=token_limit

    def lookup(self, prompt: str, llm_string: str) -> Optional[List[Generation]]:
        """Look up based on prompt and llm_string."""
        promptwatch_context = PromptWatch.get_active_instance()
        cache_prompt_key = f"{llm_string}\n:{prompt}"
        
        if promptwatch_context:
            cache = promptwatch_context.caching.get_or_init_cache(self.cache_namespace_key, self.embed_func, self.token_limit, self.similarity_limit)
            cached_res=cache.get(cache_prompt_key)
            
            if cached_res:         
                return  [Generation(text=cached_res.result, generation_info={"cached":True, **cached_res.metadata, "_cached_result":cached_res})]
       
            else:
                return None

    
    def update(self, prompt: str, llm_string: str, return_val: List[Generation]) -> None:
        """Update cache based on prompt and llm_string."""
        
        cached_res = return_val[0].generation_info.get("_cached_result")

        
        promptwatch_context = PromptWatch.get_active_instance()
        cache = promptwatch_context.caching.get_or_init_cache(self.cache_namespace_key, self.embed_func, self.token_limit, self.similarity_limit)
        cache.add(cached_res)




    
    def clear(self, until:datetime=None) -> None:
        promptwatch_context = PromptWatch.get_active_instance()
        if promptwatch_context:
            cache = promptwatch_context.caching.get_or_init_cache(self.cache_namespace_key, self.embed_func, self.token_limit, self.similarity_limit)
            cache.clear()
        else:
            Client().clear_cache(self.cache_namespace_key,until=until)




