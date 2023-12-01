# fast api
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Key값을 가져오기 위함
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
# 텍스트 생성 관련 라이브러리 모음
import openai
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent
# 전처리 관련 라이브러리 모음
import re
import pandas as pd
from function import *

import time

# FastAPI
app = FastAPI(debug = True)
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

# get 요청
@app.get("/game/") # url 예시: /game?grade=3&sem=1&type=sisa&game=context
def main(grade: str, sem: str, game: str, type: str):
    # grade: number; 학년 # sem: number; 학기 # game: 'context' | 'word'; 문맥추론, 어휘 # type: 'text' | 'sisa'; 교과서 데이터, 시사 데이터 
    if type == 'sisa':
        kind_of_data = "시사" 
    elif type == 'text':
        kind_of_data = "교과서" 
    
    if game == 'context':
        kind_of_game = "문맥추론" 
    elif game == 'word':
        kind_of_game = "어휘"
    
    test_list = ['금일', '사흘', '낭송', '사서', '고지식', '설빔']
    joined_str = ', '.join(test_list)

    # 1,4 -> 문맥추론 Template(시사, 교과서)
    # 3,5 -> 어휘 Template(시사, 교과서)
    # 2 -> Json Template
    prompt_template_1 = read_template('./prompt_templates/prompt_template_1.txt')
    prompt_template_2 = read_template('./prompt_templates/prompt_template_2.txt')
    prompt_template_3 = read_template('./prompt_templates/prompt_template_3.txt')
    prompt_template_4 = read_template('./prompt_templates/prompt_template_4.txt')
    prompt_template_5 = read_template('./prompt_templates/prompt_template_5.txt')

    memory = ConversationBufferMemory(memory_key="chat_history")
    llm=OpenAI(temperature=0.8,model_name="gpt-3.5-turbo-0613")

    if kind_of_data == "시사" and kind_of_game == "문맥추론":
        try:
            search = SerpAPIWrapper()

            tools = [
            Tool(
                name = "News Search",
                func=search.run,
                description="useful for when you need to answer questions about Recent Current Events or News in South Korea, Searching on Korea Website"
            ),
            ]

            agent_chain = initialize_agent(tools, llm, AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, memory=memory)
            refined_words_list = generate_issue_words(agent_chain, grade)
            prompt1 = make_issue_prompt(prompt_template_1,refined_words_list,grade)
            final_json_response = create_questions_and_convert_to_json(prompt1, prompt_template_2)

            return final_json_response
        
        except Exception as e:
            return None

    elif kind_of_data == "시사" and kind_of_game == "어휘":
        try:
            search = SerpAPIWrapper()

            tools = [
            Tool(
                name = "News Search",
                func=search.run,
                description="useful for when you need to answer questions about Recent Current Events or News in South Korea, Searching on Korea Website"
            ),
            ]

            agent_chain = initialize_agent(tools, llm, AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, memory=memory)
            refined_words_list = generate_issue_words(agent_chain, grade)
            prompt1 = make_issue_prompt(prompt_template_3,refined_words_list,grade)
            final_json_response = create_questions_and_convert_to_json(prompt1, prompt_template_2)

            return final_json_response
        
        except Exception as e:
            return None
        
    elif kind_of_data == "교과서" and kind_of_game == "문맥추론":
        try:
            textbook_words = filter_by_grade_sem(grade, sem)
            textbook_words_list = make_textbook_words_list(grade,sem,textbook_words,prompt_template_4)
            final_json_response = create_questions_and_convert_to_json(textbook_words_list, prompt_template_2)

            return final_json_response
        
        except Exception as e:
            return None
    else:
        try:
            textbook_words = filter_by_grade_sem(grade, sem)
            textbook_words_list = make_textbook_words_list(grade,sem,textbook_words,prompt_template_5)
            final_json_response = create_questions_and_convert_to_json(textbook_words_list, prompt_template_2)

            return final_json_response
        
        except Exception as e:
            return None

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host = "0.0.0.0", port = 8000)