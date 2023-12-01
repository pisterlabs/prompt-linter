from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Iterable, AsyncGenerator, AsyncIterable
import pinecone
import json
import openai
import asyncio
from uuid import uuid4
import time
import os
import logging
import sys
from llama_index.llms import OpenAI
from llama_index.indices.service_context import ServiceContext
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
from llama_index import (
    KnowledgeGraphIndex,
    LLMPredictor,
    ServiceContext,
    SimpleDirectoryReader,
)
from llama_index.storage.storage_context import StorageContext
from llama_index.graph_stores import NebulaGraphStore

from llama_index.llms import OpenAI
from IPython.display import Markdown, display
from llama_index import Document, VectorStoreIndex
from serpapi import GoogleSearch
import ast
import os
import requests
import Axis_prompts
openai.api_key = "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"
app = FastAPI()

origins = [
    "*",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CompletionGenerator:

    def generate_completion(prompt: str,query: str) -> str:
        completion = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          temperature = 0.7,
          max_tokens=256,
          messages=[
            {"role": "assistant", "content": prompt},
            {"role": "user", "content": query}
          ]
        )
        return completion.choices[0].message

    async def generate_acompletion(self, prompt: str,query: str) -> str:
        completion = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            temperature = 0.7,
            max_tokens=256,
            messages=[
            {"role": "assistant", "content": prompt},
            {"role": "user", "content": query}
          ]
        )
        return completion.choices[0].message

    def generate_stream_completion(self, prompt: str) -> Iterable[str]:
        stream_response = client.completion_stream(
            prompt=f"{anthropic.HUMAN_PROMPT}{prompt}{anthropic.AI_PROMPT}",
            stop_sequences=[anthropic.HUMAN_PROMPT],
            max_tokens_to_sample=self.max_tokens,
            model="claude-v1",
            stream=True,
        )
        current_completion = ''
        for data in stream_response:
            delta = data["completion"][len(current_completion):]
            current_completion = data["completion"]
            print(delta, end='', flush=True)
            yield delta



fields = [
"ACADEMIC QUALIFICATION",
"JOB EXPERIENCE",
"PROJECTS",
"POSITION OF RESPONSIBILITY",
"ACHIEVEMENTS AND AWARDS",
"PUBLICATIONS",
"COURSE WORK",
"SKILLS AND EXPERTISE",
"EXTRACIRRICULAR ACTIVITIES"
]

class CompletionRequest(BaseModel):
    user_id: str
    JD_id: str = None
    CV: str = None
    JD: str = None

    
class CV_database(CompletionGenerator):
    def __init__(self, userId: str):
        self.CV = None
        self.CV_new = None
        self.rating_section = None

    def upload_CV(self, CV):
        self.CV = CV
        return "CV uploaded Successfully"


    def reformat_CV(CV):
        CV_info = {}
        for section in fields:
            CV_conversion_prompt = Axis_prompts.CV_converter_template(CV,section)
            formated_CV = CompletionGenerator.generate_completion(prompt=CV_conversion_prompt,query="")
            CV_info[section] =formated_CV["content"]
            print(f"done with section {section}")
        return CV_info

    def calc_eval(rating_section):
        avg_score_lst = []
        for key, value in rating_section.items():
            avg_section_score = (value['Clarity and Conciseness']['Score'] + value['Relevance']['Score'] + value['Depth of Experience']['Score'])/3
            avg_score_lst.append(f"Avg score: {key} -> {avg_section_score}")
        return avg_score_lst

    def eval_CV_JD(self, JD):
        self.CV_new = CV_database.reformat_CV(self.CV)
        print(self.CV,self.CV_new,JD)
        rating_section = {}
        for section, section_content in self.CV_new.items():
            rating_prompt = Axis_prompts.rating_template(JD,section,section_content) 
            rating = CompletionGenerator.generate_completion(prompt=rating_prompt,query="")
            rating_string =rating["content"]
            rating_string.replace("\n", "").replace(" ", "")
            rating_dictionary = ast.literal_eval(rating_string)
            rating_section[section] = rating_dictionary
        self.rating_section = rating_section
        rating = CV_database.calc_eval(self.rating_section)
        return(rating)


class JD_database(CompletionGenerator):
    def __init__(self, userId: str):
        self.JD = None

    def upload_JD(self, JD):
        self.JD = JD
        return "JD uploaded Successfully"


CV_database_generator = {}
JD_database_generator = {}

@app.post("/upload-CV")
def call_CV(request: CompletionRequest):
    user_id = request.user_id
    CV = request.CV
    print("reached here")
    CV_database_generator[user_id] = CV_database(user_id)
    CV_upload = CV_database_generator[user_id].upload_CV(CV)
    return JSONResponse({"data": {},"status": 200, "message": CV_upload})


@app.post("/upload-JD")
def call_JD(request: CompletionRequest):
    JD_id = request.JD_id
    JD = request.JD
    JD_database_generator[JD_id] = JD_database(JD_id)
    JD_upload = JD_database_generator[JD_id].upload_JD(JD)
    return JSONResponse({"data": {},"status": 200, "message": JD_upload})

@app.post("/eval-CV")
def eval_CV(request: CompletionRequest):
    JD_id = request.JD_id
    user_id = request.user_id
    JD = JD_database_generator[JD_id].JD
    CV_eval = CV_database_generator[user_id].eval_CV_JD(JD)
    return JSONResponse({"data": CV_eval,"status": 200, "message": "CV evaluted successfully"})