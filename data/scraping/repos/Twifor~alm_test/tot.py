from tkinter import E
from turtle import begin_fill
from agent.tools import ToolList
from examples.bmbtools.douban import (
    PrintDetailTool,
    NowPlayingOutFilterTool,
    ComingOutFilterTool,
)
from examples.hotpotQA.hotpotqa_tools import EM
import json
import random
from examples.bmbtools.weather import (
    ForcastWeatherTool,
    GetWeatherTool,
)
from examples.bmbtools.file_operation import WriteFileTool, ReadFileTool
from examples.bmbtools.wikipedia import (
    WikiPediaSearchTool,
    WikiLookUpTool,
    WikiPediaDisambiguationTool,
)
from examples.bmbtools.answer import AnswerTool, BeginTool
from examples.bmbtools.python import RunPythonTool
from examples.bmbtools.google_search import GoogleSearchTool, GoogleSearch2Tool
from examples.bmbtools.code_interpreter import ExecuteCodeTool
from examples.bmbtools.gradio import ImageCaptionTool, ImageToPromptTool, OCRTool
from examples.scienceQA.read_lecture import ReadLectureTool
from examples.scienceQA.rubbish_tools import *
from agent.alm import ReActAgent, ToTAgent
from agent.llm import GPT3_5LLM, Davinci003LLM
from utils.loadenv import Env
import openai
import json

env = Env()
llm = GPT3_5LLM(env.openai_key())
tools = [
    BeginTool(),
    AnswerTool(lambda x: True),
    WikiPediaSearchTool(),
    WikiLookUpTool(),
    WikiPediaDisambiguationTool(),
    GoogleSearch2Tool(),
    OCRTool(),
    # ImageCaptionTool(),
    GoogleSearchTool(env.searper_key()),
    ExecuteCodeTool(),
    RunPythonTool(),
    R_OCRTool(),
    R_SearchTool(),
    R_UnknownTool(),
    R_LoopUpTool(),
    ReadLectureTool(""),
]
tool_list = ToolList()
for tool in tools:
    tool_list.registerTool(tool)

i = 0
while i < 1000:
    try:
        react_agent = ToTAgent(llm, tool_list)
        file = open(f"dataset/scienceQA/test/{i}.json", "r")
        obj = json.loads(file.read())
        query = (
            obj["question"]
            + " You must choose one answer from the following choices:\n"
        )
        query += "Choices: " + str(obj["choices"]) + "\n"
        if "image" in obj.keys():
            query += "This question has a related image, you can use some tools to read from this image to help you to solve this problem.\n"
            query += f"The path of this image: dataset/scienceQA/train/{i}.jpg."
        ans = obj["choices"][obj["answer"]]
        lecture = obj["lecture"]

        tools[1].func = lambda x: EM(x, ans)
        tools[-1].knowledge = lecture
        react_agent.setRequest(query)
        llm.tokens = 0
        react_agent.steps(max_steps=8)
        import os

        os.system(f"del ./logs/tot_sciQA_{i}.log")
        react_agent.saveLog(
            f"tot_sciQA_{i}", {"ground_truth": ans, "token_use": llm.tokens}
        )
        i += 1
    except:
        continue
