#!/usr/bin/env python
# coding: utf-8

import json

import pandas as pd
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from tqdm import tqdm

# langchain setup
MODEL_PATH = "/Users/ian/.cache/lm-studio/models/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/mistral-7b-instruct-v0.1.Q8_0.gguf"
template = """<s>[INST] {human_input} [/INST]"""
prompt = PromptTemplate(input_variables=["human_input"], template=template)
prompt_template = """Output valid JSON as your response. Please read the follwing text then fill in the provided JSON data with answers to each question. Make logical assumptions when necessary to fill in as many answers as possible.

Request:
{text}

Response:
{question}
"""

# read data files
transcripts = json.load(
    open("/Users/ian/Downloads/purdue-data-for-good-2023/transcripts.json")
)
test = pd.read_csv("/Users/ian/Downloads/purdue-data-for-good-2023/test.csv")

# initialize langchain
llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_gpu_layers=1,
    n_batch=512,
    n_ctx=4096,
    f16_kv=True,
    max_tokens=4096,
    # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    callback_manager=None,
    verbose=False,
    temperature=0,
)
chat_llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=False,
)


def store_result(data):
    try:
        pd.DataFrame(data).to_csv("results.csv", index=False, mode="a", header=False)
    except ValueError:
        print(data)


def get_answer_by_transcript_id(transcript_id: int) -> list:
    questions = test[test["Transcript"] == int(transcript_id)].to_dict(orient="records")
    for question in questions:
        question["Answer"] = ""
    p = prompt_template.format(
        text=transcripts[str(transcript_id)], question=json.dumps(questions)
    )
    res = chat_llm_chain.predict(human_input=p)
    try:
        res_dict = json.loads(res)
        return res_dict
    except json.decoder.JSONDecodeError:
        print(f"[transcript_id: {transcript_id}] Error decoding JSON: {res}")
        return []


# res = get_answer_by_transcript_id(291)
# print(res)

start_after = 3728
is_start = False
transcript_ids = list(transcripts.keys())
for transcript_id in tqdm(transcript_ids):
    if int(transcript_id) == start_after:
        is_start = True
        continue
    if not is_start:
        print(f"Skipping transcript_id: {transcript_id}, looking for {start_after}")
        continue
    res = get_answer_by_transcript_id(transcript_id)
    store_result(res)
