"""
    Hugging face model loader
"""

from abc import ABC, abstractmethod
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


class BaseModelLoader(ABC):
    """Abstract base class for loading models."""

    def __init__(self, task: str):
        self.task = task
        self.model = None
        self.tokenizer = None
        self.pipeline = None

    @property
    def task(self):
        return self._task

    @task.setter
    def task(self, val: str):
        allowed_tasks = ["translate", "recognize"]

        if val not in allowed_tasks:
            raise ValueError(f"Expected task to be eiter {', '.join(allowed_tasks)}"
                             f"but got {val}")

        self._task = val

    @abstractmethod
    def load_model(self, **kwargs):
        """Load the model."""
        pass

    @abstractmethod
    def load_tokenizer(self, **kwargs):
        """Load the tokenizer."""
        pass

    @abstractmethod
    def load_pipeline(self, **kwargs):
        """Load the processing pipeline."""
        pass

    @abstractmethod
    def generate(self, **kwargs):
        """Generate output based on input."""
        pass


class OpenAIModelLoader(BaseModelLoader):
    """Loads models for translation and recognition using OpenAI's GPT-3."""
    def __init__(self, task: str, **kwargs):
        super().__init__(task)

        self.prompt = PromptTemplate.from_template(
            """
                You're a truthful assistant who {task} languages correctly else reply 'failed'
                
                {task} {add_task}: {sentence}
            """
        )

        self.model = self.load_model(**kwargs)
        self.pipeline = self.load_pipeline()

    def load_model(self, **kwargs):
        """Load the model and return from ChatOpenAI(**kwargs)"""
        return ChatOpenAI(**kwargs)

    def load_tokenizer(self):
        """Load the tokenizer."""
        return None

    def load_pipeline(self):
        """Load the processing pipeline."""
        return LLMChain(llm=self.model, prompt=self.prompt)

    def generate(self, inputs: dict, **kwargs):
        """Generate output based on input."""
        return self.pipeline.run(task=self.task, add_task=inputs["add_task"],
                                 sentence=inputs["sentence"])


class LlamaGPTQ(BaseModelLoader):
    """Loads models for various language tasks using Hugging Face's transformers library."""

    def __init__(self, task):
        super().__init__(task)

        self.prompt = PromptTemplate.from_template(
            """[INST] <<SYS>>
                You're a truthful assistant who {task} languages correctly else reply 'failed'
                <</SYS>>
                {task} {add_task}: {sentence}
                [/INST]
            """
        )

    def load_model(self, model_name_or_path: str, revision: str = "main"):
        return AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            device_map="auto",
            revision=revision
        )

    def load_tokenizer(self, model_name_or_path, **kwargs):
        return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    def load_pipeline(self, max_new_tokens: int = 512, temperature: int = 0.7,
                      top_p: int = 0.95):
        return pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature
        )

    def generate(self, inputs: dict, **kwargs):
        """Generate output based on input."""
        prompt = self.prompt.format(task=self.task, add_task=inputs["add_task"],
                                    sentence=inputs["sentence"])
        return self.pipeline(prompt)


if __name__ == "__main__":
    # test openAI run
    from config import openai_key
    import os
    os.environ["OPENAI_API_KEY"] = openai_key
    llm = OpenAIModelLoader("translate")
    print(llm.generate(
        {
            "add_task": "to hinglish",
            "sentence": "who is that person that won't give up training at the brink of death"
        }
    ))
