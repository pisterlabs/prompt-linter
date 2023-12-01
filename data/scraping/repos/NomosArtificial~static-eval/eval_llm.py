import fire
import json
import time
import math
import asyncio
import torch
from typing import Iterable, Dict
from tqdm import tqdm
from dataclasses import dataclass
from langchain.llms.base import BaseLLM
from langchain.llms import OpenAI, Anthropic
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from custom_hf_llm import CustomHFModel
from utils import chunked_iterable


@dataclass
class LLMConfig:
    model_name: str
    max_new_tokens: int
    temperature: float
    rate_limit: int = None


SUPPORTED_MODELS = [
    "gpt-3.5-turbo",
    "gpt-4",
    "text-davinci-002",
    "text-davinci-003",
    "claude-instant-v1",
    "claude-v1.2",
    "claude-v1.3",
]

RATE_LIMITS = {
    "gpt-3.5-turbo": 300,
    "gpt-4": 200,
    "text-davinci-002": 3250,
    "text-davinci-003": 3250,
    "claude-instant-v1": None,
    "claude-v1.2": None,
    "claude-v1.3": None,
}


"""
Wrapper for running LangChain models on a set of inputs with a simplified interface that handles a few things:
* asynchronous requests for supported LLM providers
* Rate limiting requests
* Backoff/retry for failed requests
* Unified template interface for Chat and vanilla LLMs based on a single string template
"""


class EvalLLM:
    def __init__(self, config: LLMConfig, template_string: str):
        self.config = config
        self.recommended_chunk_size = 20
        if self.config.rate_limit is None:
            self.config.rate_limit = RATE_LIMITS[config.model_name]

        self.template_string = template_string
        self.is_chat = config.model_name in ["gpt-3.5-turbo", "gpt-4"]
        self.apply_async = config.model_name in [
            "gpt-3.5-turbo",
            "gpt-4",
            "text-davinci-002",
            "text-davinci-003",
        ]
        base_prompt = PromptTemplate(
            template=template_string,
            template_format="jinja2",
            input_variables=["document"],
            validate_template=False,
        )
        prompt = None

        if self.is_chat:
            human_message_template = HumanMessagePromptTemplate(prompt=base_prompt)
            prompt = ChatPromptTemplate.from_messages([human_message_template])
        else:
            prompt = base_prompt

        model = None

        if config.model_name in ["gpt-3.5-turbo", "gpt-4"]:
            model = ChatOpenAI(
                model_name=config.model_name,
                max_tokens=config.max_new_tokens,
                temperature=config.temperature,
            )
        elif config.model_name in ["claude-instant-v1", "claude-v1.2", "claude-v1.3"]:
            model = Anthropic(
                model=config.model_name,
                max_tokens_to_sample=config.max_new_tokens,
                temperature=config.temperature,
            )
        elif config.model_name in ["text-davinci-002", "text-davinci-003"]:
            model = OpenAI(
                model_name=config.model_name,
                max_tokens=config.max_new_tokens,
                temperature=config.temperature,
            )
        elif "LawInformedAI/" in config.model_name:
            checkpoint = config.model_name
            weights_location = hf_hub_download(
                checkpoint, "pytorch_model.bin.index.json"
            )
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)

            with init_empty_weights():
                if "t5" in config.model_name:
                    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
                elif "gptj" in config.model_name or "llama" in config.model_name:
                    model = AutoModelForCausalLM.from_pretrained(checkpoint)
                else:
                    raise NotImplementedError

            model = load_checkpoint_and_dispatch(
                model=model,
                checkpoint=weights_location,
                dtype=torch.bfloat16,
                device_map="auto",
            )
            model_kwargs = {
                "temperature": config.temperature,
                "max_new_tokens": config.max_new_tokens,
            }
            model = CustomHFModel(
                model=model, tokenizer=tokenizer, generate_kwargs=model_kwargs
            )
            self.recommended_chunk_size = 4
        self.chain = LLMChain(llm=model, prompt=prompt)

    async def run(self, inputs: Iterable[Dict], chunk_size: int = 20):
        start = time.time()
        completions = []
        chain_input = [{"document": input} for input in inputs]
        print(
            "Generating completions with rate limit of {} calls per minute".format(
                self.config.rate_limit
            )
        )
        for chunk_idx, chunk in enumerate(
            tqdm(
                chunked_iterable(chain_input, chunk_size),
                total=math.ceil(len(chain_input) / chunk_size),
            )
        ):
            if self.apply_async:
                completions += await self.chain.aapply(chunk)
            else:
                completions += self.chain.apply(chunk)

            if self.config.rate_limit is None:
                continue
            # check if we are over the rate limit
            elapsed = time.time() - start
            total_calls = (chunk_idx + 1) * (chunk_size + 1)
            expected_time = total_calls * 1 / self.config.rate_limit * 60
            await asyncio.sleep(0.5)
            if (
                elapsed < expected_time + 15
                and chunk_idx < math.ceil(len(chain_input) / chunk_size) - 1
            ):
                # wait for elapsed time to catch up with expected time for current number of calls
                await asyncio.sleep(expected_time - elapsed + 15)
        print(
            "Finished generating completions in {:.2f} seconds".format(
                time.time() - start
            )
        )
        return completions


def try_parse(completion):
    try:
        result = json.loads(completion)
        return result["answer"]
    except:
        return ""


async def test_eval_llm():
    questions = list(load_dataset("andersonbcdefg/mpre", split="train"))
    answers = ["abcd"[q["correct_idx"]] for q in questions]
    template_string = """You are a legal expert who specializes in professional ethics. \
In response to the following multiple-choice question about legal professional ethics, please identify the correct answer. \
Your response should be in JSON with a single key, "answer", and a value of "a", "b", "c", or "d".

QUESTION: {{ document.problem_statement }}

OPTIONS:
a. {{ document.candidate_answers[0] }}
b. {{ document.candidate_answers[1] }}
c. {{ document.candidate_answers[2] }}
d. {{ document.candidate_answers[3] }}

ANSWER:"""

    for model_name, rate_limit in zip(
        SUPPORTED_MODELS, [3500, 200, 3500, 3500, None, None]
    ):
        config = LLMConfig(
            model_name=model_name, max_new_tokens=20, rate_limit=rate_limit
        )
        llm = EvalLLM(config, template_string)
        completions = await llm.run(questions)
        preds = [try_parse(c["text"]) for c in completions]
        print(
            f"{model_name} Accuracy:",
            sum([p == a for p, a in zip(preds, answers)]) / len(answers),
        )


if __name__ == "__main__":
    fire.Fire(test_eval_llm)
