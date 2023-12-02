# ADAPTED FROM: https://github.com/noahshinn024/reflexion/blob/main/programming_runs/generators/model.py
from typing import List

from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage

from .base import ChatLLM


class StarChat(ChatLLM):
    MESSAGE_TO_TOKEN = {
        HumanMessage: "<|user|>",
        AIMessage: "<|assistant|>",
        SystemMessage: "<|system|>",
    }

    def __init__(self, max_tokens: int = 1024, temperature: float = 0.2):
        try:
            import torch
            from transformers import pipeline
        except ImportError:
            raise ImportError(
                "Please install transformers and torch to use the StarChat LLM."
            )
        self.name = "star-chat"
        self.pipe = pipeline(
            "text-generation",
            model="HuggingFaceH4/starchat-beta",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.max_tokens = max_tokens
        self.temperature = max(temperature, 1e-4)  # NOTE: HF does not like temp of 0.0.

    def __call__(self, messages: List[BaseMessage]) -> AIMessage:
        prompt = "".join(
            f"{self.MESSAGE_TO_TOKEN[type(message)]}\n{message.content}\n<|end|>\n"
            for message in messages
        )
        prompt += "<|assistant|>\n"

        outputs = self.pipe(
            prompt,
            max_new_tokens=self.max_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=0.95,
            eos_token_id=49155,
            num_return_sequences=1,
        )

        outs = (
            output["generated_text"].split("<|assistant|>")[1].rstrip("<|end|>")
            for output in outputs
        )

        return AIMessage(content=next(outs))
