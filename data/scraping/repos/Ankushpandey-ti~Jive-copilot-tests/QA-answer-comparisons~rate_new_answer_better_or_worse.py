from typing import Dict
import json
from langchain.llms import PromptLayerOpenAI
import promptlayer
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import openai
import os
from dotenv import load_dotenv
load_dotenv()
import multiprocessing

PROMPT_LAYER_API_KEY= os.environ['PROMPT_LAYER_API_KEY']
OPEN_API_KEY=os.environ['OPENAI_API_KEY']
FILE_PATH= 'question_search_phrase_sources.json'

def get_prompts(question:str,old_answer:str,new_answer:str):
    system_message_prompt = f'''You are an AI assistant that tells if the new_answer is better than previous answer for this given question: {question}. Old_answer:{old_answer}, New_answer:{new_answer} and categorize the question in one of the provided category.'''
    human_message_prompt = f'''
Compare the two answers provided for this question: {question} and provide output in this json format.

Follow these instructions while providing output.
1. new_answer_quality can be BETTER,WORSE or SIMILAR 
2. question_type can be categorised as : GENERIC_QUESTION , LATEST_DATA_BASED_QUESTION, AGGREGATION_QUESTION, SUBJECTIVE_QUESTION
3. Output should be in this json format - 
---
    {{
    "new_answer_quality": "new answer quality ",
    "reason": "Reason for new_answer_quality",
    "question_type": "question type out of the provided question categories"
    ""
    }}
----
'''
    return {'human_message_prompt':human_message_prompt,'system_message_prompt':system_message_prompt}



def get_data_from_file(file_path) -> Dict:
    with open(file_path, 'r') as file:
        json_data = json.load(file)

    json_object = json_data
    # Now you can work with the JSON object
    return json_object

def write_to_file(filepath, file_data):
    with open(filepath, 'w') as file:
        file.write(json.dumps(file_data))

    print(f"Data written to {filepath}")
    return

def rate_new_answer_quality():
    promptlayer.api_key = PROMPT_LAYER_API_KEY
    file_path ='./final_data_file.json'
    file_data = get_data_from_file(file_path)

    print(f'Read data from file: {file_path}')
    file_new_data =[]

    ## Iterating over the data in file
    for data in file_data:
        try:
            single_question_data =data
            new_answer = data.get('search_phrase_answer')
            question = data.get('question')
            old_answer = data.get('original')
        except Exception as e:
            print(f'Exception occured while reading the data from file. Exception:{e}')

        
        try:
            prompts_content = get_prompts(new_answer=new_answer,old_answer=old_answer,question=question)
            messages = [
                {'role':'system','content':prompts_content['system_message_prompt']},
                {'role':'user','content':prompts_content['human_message_prompt']}
            ]
            openai.api_key = OPEN_API_KEY
            # user_message_for_quick_action_summary(quick_action_json)
            summary = openai.ChatCompletion.create(model='gpt-3.5-turbo-16k', messages=messages,temperature=0.4)  # this time, we set stream=True):        
            response = summary['choices'][0]['message']['content']
            # chat = PromptLayerOpenAI(temperature=0.4, openai_api_key=OPEN_API_KEY,verbose=True,
            #                         streaming=False, pl_tags=["answer-similarity"])
            # print('Here')
            # resp = chat.predict_messages([HumanMessage(content=prompts_content['human_message_prompt']),
            #                                         SystemMessage(content=prompts_content['system_message_prompt']),
            #                                         AIMessage(content='You are an AI assisstant that compares new answer with old answer and tells if its is better or worse based for a given question and provide a category to the question. You always provide output in json format ')
            #                                         ])
            print('AFTER_GPT response')
            response = json.loads(response,strict=False)
            print(f'CHAT_GPT_response: {response}')
            single_question_data['search_phrase_answer_category'] = response.get('new_answer_quality')
            single_question_data['search_phrase_answer_category_reason'] = response.get('reason')
            single_question_data['question_category'] = response.get('question_type')

        except Exception as e:
            print(f'Exception Occured: {e}')
            single_question_data = data

        file_new_data.append(single_question_data)

    write_to_file('new_answer_categorised.json', file_new_data)

    

rate_new_answer_quality()

