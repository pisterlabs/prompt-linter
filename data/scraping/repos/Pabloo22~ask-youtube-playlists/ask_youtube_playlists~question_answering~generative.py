"""Contains the functionality to answer a question using generative
models."""
import streamlit as st
from dataclasses import dataclass
from typing import List, Dict, Union

import langchain
from langchain import llms
from langchain.schema import Document


@dataclass
class LLMSpec:
    """Class to store the information of a language model.

    Attributes:
        model_name (str): The name of the language model.
        model_type (str): The class or method used to load the language model.
    """
    model_name: str
    model_type: str
    max_tokens: int


GENERATIVE_MODELS = [
    LLMSpec("gpt2", "huggingface-pipeline", max_tokens=1024),
    LLMSpec("gpt-3.5-turbo", "openai-chat", max_tokens=4096),
    LLMSpec("gpt-3.5-turbo-16k", "openai-chat", max_tokens=16384),
    LLMSpec("gpt-4", "openai-chat", max_tokens=8192),
]

GENERATIVE_MODEL_NAMES = [model_spec.model_name
                          for model_spec in GENERATIVE_MODELS]


def get_model_spec(model_name: str) -> LLMSpec:
    """Returns the language model specification.

    Args:
        model_name (str): The name of the language model.

    Returns:
        LLMSpec: The language model specification.

    Raises:
        ValueError: If the language model is not available.
    """
    for model_spec in GENERATIVE_MODELS:
        if model_spec.model_name == model_name:
            return model_spec

    raise ValueError(f"Model '{model_name}' not available. Available "
                     f"models are: {GENERATIVE_MODELS}")


def load_model(model_name: str,
               temperature: float = 0.7,
               max_length: int = 1024,
               ) -> llms.base.BaseLLM:
    """Loads the language model.

    Args:
        model_name (str): The language model name.
        temperature (float, optional): The temperature used to generate the
            answer. The higher the temperature, the more "creative" the answer
            will be. Defaults to 0.7.
        max_length (int, optional): The maximum length of the generated answer.
            Defaults to 128.

    Returns:
        llms.base.BaseLLM: The language model.
    """

    model_spec = get_model_spec(model_name)

    if model_spec.model_type == "openai-chat":
        llm = llms.OpenAIChat(  # type: ignore
            model_name=model_spec.model_name,
            model_kwargs={"temperature": temperature,
                          "max_length": max_length}
        )
        return llm
    if model_spec.model_type == "huggingface-pipeline":
        llm = llms.HuggingFacePipeline.from_model_id(  # type: ignore
            model_id=model_spec.model_name,
            task="text-generation",
            model_kwargs={"temperature": temperature,
                          "max_length": max_length}
        )
        return llm
    available_models = [model.model_name for model in GENERATIVE_MODELS]
    raise ValueError(f"Model type '{model_spec.model_type}' not available. "
                     f"Available models are: {available_models}")


def _get_generative_prompt_template(retrieved_documents: List[Document],
                                    ) -> langchain.PromptTemplate:
    """Returns the template used to generate the answer.

    Returns:
        langchain.PromptTemplate: The template used to generate the answer.
    """
    template_text = ""
    for document in reversed(retrieved_documents):
        template_text += f"{document.page_content}\n\n"

    template_text += "Question: {question}\n\n"
    template_text += "Answer:"

    template = langchain.PromptTemplate(template=template_text,
                                        input_variables=["question"])

    return template


@st.cache_data
def get_generative_answer(question: str,
                          relevant_documents: List[Document],
                          model_name: str,
                          temperature: int,
                          max_length: int) -> str:
    """Returns the answer to the question as a string.

    Args:
        question (str): The question asked by the user.
        relevant_documents (List[Document]): The list of relevant documents.
        model_name (str): The name of the language model.
        temperature (float): The temperature used to generate the answer.
        max_length (int): The maximum length of the generated answer.

    Returns:
        str: The answer to the question.
    """
    model = load_model(model_name=model_name,
                       temperature=temperature,
                       max_length=max_length)
    template = _get_generative_prompt_template(relevant_documents)
    prompt = template.format(question=question)
    answer = model.generate(prompts=[prompt])
    return answer.generations[0][0].text
