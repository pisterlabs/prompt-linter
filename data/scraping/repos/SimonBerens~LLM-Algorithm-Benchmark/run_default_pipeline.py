import asyncio
import json

from langchain import OpenAI, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

import paths
from execution_pipeline.code_exectuors.python_executor import PythonExecutor
from execution_pipeline.evaluators.stripped_exact_match import StrippedExactMatch
from execution_pipeline.languages import SupportedLanguage
from execution_pipeline.llm_executors.python_langchain import PythonLangchainExecutor
from execution_pipeline.pipelines.default_pipeline import DefaultPipeline
from execution_pipeline.prompting.prompts import *
from execution_pipeline.task_runners.default_task_runner import DefaultTaskRunner
from execution_pipeline.types import Executor
from utils import force_write_to
from writer import get_json_test_results

models = [
    {
        "model_name": "text-davinci-002",
        "is_chat_model": False,
    },
    {
        "model_name": "text-davinci-003",
        "is_chat_model": False,
    },
    {
        "model_name": "gpt-3.5-turbo",
        "is_chat_model": True,
    },
    {
        "model_name": "gpt-4",
        "is_chat_model": True,
    },
]

python_llm_executors: list[Executor] = []


for model in models:
    if model["is_chat_model"]:
        prompts = list(map(ChatPromptTemplate.from_messages, [zero_shot_messages, few_shot_messages]))
        llm = ChatOpenAI(temperature=0, model_name=model["model_name"])
    else:
        prompts = [zero_shot_template, few_shot_template]
        llm = OpenAI(temperature=0, model_name=model["model_name"])
    for i, prompt in enumerate(prompts):
        name = model["model_name"] + "_" + ("zero_shot" if i == 0 else "few_shot")
        executor = PythonLangchainExecutor(name=name, langchain=LLMChain(llm=llm, prompt=prompt))
        python_llm_executors.append(executor)


code_executor_mapping = {SupportedLanguage.PYTHON: PythonExecutor()}
llm_executor_mapping = {SupportedLanguage.PYTHON: python_llm_executors}

task_runner = DefaultTaskRunner(code_executor_mapping, llm_executor_mapping, StrippedExactMatch())

print("Running pipeline...")
results = asyncio.run(DefaultPipeline(task_runner).run())
print("Writing results...")
path = paths.root_path.joinpath('results/result.json')
force_write_to(path, json.dumps(get_json_test_results(results)))
print("Done!")
