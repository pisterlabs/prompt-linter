import os
from typing import List

import anthropic
from anthropic._tokenizers import sync_get_tokenizer as get_tokenizer

from ..llmtypes import (
    LLMCaller, LLMCallArgs, AnthropicModelType, LLMModelType, LLMMessage
)

class ClaudeCaller(LLMCaller):
    def __init__(
        self, model: str = "claude-2"
    ):
        Modtype = LLMModelType.get_type(model)
        if isinstance(Modtype, tuple):
            raise ValueError(f"{model} is ambiguous type")
        if Modtype not in [AnthropicModelType]:
            raise ValueError(f"{model} is not supported")
        
        model_: LLMModelType = Modtype(Name=model)
        if model_.Client_Args.get("api_key") is None:
            model_.Client_Args["api_key"] = os.environ.get("ANTHROPIC_KEY")
        
        if Modtype in [AnthropicModelType]:
            client = anthropic.Client(**model_.Client_Args)
            aclient = anthropic.AsyncClient(**model_.Client_Args)
            
        super().__init__(
            Model = LLMModelType(Name=model),
            Func = client.completions.create,
            AFunc = aclient.completions.create,
            Args = LLMCallArgs(
                Model="model", Messages="prompt", Max_Tokens="max_tokens_to_sample"
            ),
            Defaults = {
                "stop_sequence": [anthropic.HUMAN_PROMPT],
            },
            Token_Window = model_.Token_Window,
            Token_Limit_Completion=model_.Token_Limit_Completion
        )
    
    def format_message(self, message: LLMMessage):
        if message.Role in ["user", "system"]:
            return f"{anthropic.HUMAN_PROMPT} {message.Message}{anthropic.AI_PROMPT}"
        elif message.Role == "assistant":
            return f"{anthropic.AI_PROMPT} {message.Message}"
        else:
            return ""
    
    def format_messagelist(self, messagelist: List[LLMMessage]):
        return "".join(self.format_message(message) for message in messagelist)
    
    def format_output(self, output):
        return LLMMessage(Role="assistant", Message=output["completion"][1:])
    
    def tokenize(self, messagelist: List[LLMMessage]):
        tokenizer = get_tokenizer()
        return tokenizer.encode(self.format_messagelist(messagelist))