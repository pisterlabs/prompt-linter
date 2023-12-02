import os
import re
import openai
from typing import Optional, List, Mapping, Any

from llama_index import (LLMPredictor, ServiceContext,
                         LangchainEmbedding, PromptHelper)
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index.node_parser.simple import SimpleNodeParser

from langchain.llms.base import LLM
from langchain.embeddings.huggingface import HuggingFaceEmbeddings


def get_service_context(llm_type="custom", max_chunk_overlap=0.5,
                        max_input_size=1500, num_output=512,
                        chunk_size_limit=512, openai_api_key=None,
                        openai_api_base=None) -> ServiceContext:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["OPENAI_API_BASE"] = openai_api_base
    os.environ["MAX_INPUT_SIZE"] = str(max_input_size)
    print(f"Setting up service context with llm_type: {llm_type}")
    if llm_type == "custom":
        llm_predictor = LLMPredictor(
            llm=CustomLLM()
        )
    elif llm_type == "custom-http":
        llm_predictor = LLMPredictor(llm=CustomHttpLLM())
    else:
        raise ValueError(f"Invalid llm_type: {llm_type}")

    embed_model = LangchainEmbedding(HuggingFaceEmbeddings())

    node_parser = SimpleNodeParser(text_splitter=TokenTextSplitter(
        chunk_size=chunk_size_limit, chunk_overlap=max_chunk_overlap)
    )
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, embed_model=embed_model,
        prompt_helper=prompt_helper, node_parser=node_parser,
        chunk_size_limit=chunk_size_limit
    )
    return service_context


class CustomHttpLLM(LLM):
    model_name = "vicuna-13b"

    def model_pipeline(self, prompt: str) -> str:
        completion = openai.ChatCompletion.create(
            api_key=os.environ["OPENAI_API_KEY"],
            api_base=os.environ["OPENAI_API_BASE"],
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )
        return completion.choices[0].message.content

    def remove_html_tags(self, text):
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        print(f"{prompt}, {type(prompt)}")
        res = self.model_pipeline(str(prompt))
        try:
            return res
        except Exception as e:
            print(e)
            return "Don't know the answer"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": self.model_name}

    @property
    def _llm_type(self) -> str:
        return "custom"


class CustomLLM(LLM):
    model_name = "eachadea/vicuna-13b-1.1"
    # model_name = "lmsys/vicuna-7b-delta-v1.1"

    def __init__(self, *args, **kwargs):
        import torch
        from transformers import pipeline
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = 'cpu'

        device = torch.device(device)
        print(f"Using device: {device}")

        self.model_pipeline = pipeline("text-generation", model=self.model_name, device_map='auto',
                                       trust_remote_code=True, model_kwargs={"torch_dtype": torch.bfloat16, "load_in_8bit": True},
                                       max_length=1500)

    def remove_html_tags(self, text):
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        print(prompt, type(prompt))
        res = self.model_pipeline(str(prompt))
        print(res, type(res))
        if len(res) >= 1:
            generated_text = res[0].get("generated_text")[len(prompt):]
            return generated_text
        else:
            return "Don't know the answer"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": self.model_name}

    @property
    def _llm_type(self) -> str:
        return "custom"
