import openai
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from openai_function_call import OpenAISchema
from typing import Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import time
import streamlit as st
import re
import json

class LlmModelType(str,Enum):
    GPT4 = 'gpt-4-0613'
    GPT_3_5_TURBO = 'gpt-3.5-turbo-0613'
    GPT_3_5_TURBO_16K = 'gpt-3.5-turbo-16k-0613'
    ANTHROPIC_CLOUDE_2 = "claude-2"
    
    @classmethod
    def openAI_models(cls)->list:
        return [cls.GPT4 , cls.GPT_3_5_TURBO , cls.GPT_3_5_TURBO_16K]


    def cost(self,usage)->float:
        

        prompt_token_cost = 0
        completion_token_cost = 0
        match self:
            case LlmModelType.GPT_3_5_TURBO:
                prompt_token_cost = 0.0002
                completion_token_cost = 0.00015
            case LlmModelType.GPT4:
                prompt_token_cost = 0.003
                completion_token_cost = 0.006
            case LlmModelType.GPT_3_5_TURBO_16K:
                prompt_token_cost = 0.0003
                completion_token_cost = 0.0004
            case LlmModelType.ANTHROPIC_CLOUDE_2:
                prompt_token_cost = 0.0000
                completion_token_cost = 0.0000

        prompt_tokens = usage['prompt_tokens']
        completion_tokens = usage['completion_tokens']  
        
        return usage['prompt_tokens'] * prompt_token_cost + usage['completion_tokens'] * completion_token_cost

def extract_json(text):
    pattern = r'<(?:json|js)>(.*?)</(?:json|js)>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return None

def get_completion_anthropic(messages,
                            dataModel : Optional[OpenAISchema],  
                            model=LlmModelType, 
                            temperature=0, 
                            max_tokens=1000):
    
   
    anthropic = Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key = st.secrets["ANTHROPIC_API_KEY"],
    )

    
    prompt = f'''

    {HUMAN_PROMPT}

    '''
    for message in messages:
        prompt += "\n\n" + message["content"] 

    if dataModel is not None:
        prompt += f'''Your output must be only json code following the json schema inclosed in <json> tags.
        All of your output will be passed to deserialized to create a json object. 
        <json>
        {dataModel.openai_schema}
        </json>

        {AI_PROMPT}:

        '''
    

        #st.write(prompt)


    completion = anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=300,
        prompt=prompt,
    )
    #st.warning(completion.completion)
    json_str = extract_json(completion.completion)
    #st.success(json_str)
    
    data = json.loads(json_str)
    #st.warning(f"data: \n\n{data}")
    object = dataModel.parse_obj(data)
    #st.warning(f"{object}")
    return object
    



def get_completion_from_messages(messages,
                                 model=LlmModelType, 
                                 temperature=0, 
                                 max_tokens=1000):
    
    try:
        response = openai.ChatCompletion.create(
            model=model.value,
            messages=messages,
            temperature=temperature, 
            max_tokens=max_tokens, 
        )
    except:       
        raise

    return (response.choices[0].message["content"] , response["usage"])

def get_completion_from_function(messages,
                                 dataModel : Optional[OpenAISchema],  
                                 model=LlmModelType, 
                                 temperature=0, 
                                 max_tokens=1000):
    

    try:
       
        if dataModel is None:
            
            response = openai.ChatCompletion.create(
                model=model.value,
                messages=messages,
                temperature=temperature, 
                max_tokens=max_tokens, 
            )
        else:
            response = openai.ChatCompletion.create(
                    model=model.value,
                    functions=[dataModel.openai_schema],
                    messages=messages,
                    temperature=temperature, 
                    max_tokens=max_tokens, 
            )
        
    except:       
        raise

    return dataModel.from_response(response) , response["usage"]

async def get_completion_from_function_async(messages,
                                 dataModel,  
                                 model=LlmModelType, 
                                 temperature=0, 
                                 max_tokens=1000):
    
    start_time = time.time()

    try:
        
        
        response = await openai.ChatCompletion.acreate(
            model=model.value,
            functions=[dataModel.openai_schema],
            messages=messages,
            temperature=temperature, 
            max_tokens=max_tokens, 
        )
    except:       
        raise

    duration = time.time() - start_time
    return dataModel.from_response(response) , response["usage"] , duration

# async def aget_completion_from_messages(messages, 
#                                  model=LlmModelType, 
#                                  temperature=0, 
#                                  max_tokens=1000):
    
#     try:
#         response = openai.ChatCompletion.create(
#             model=model.value,
#             messages=messages,
#             temperature=temperature, 
#             max_tokens=max_tokens, 
#         )
#     except:       
#         raise

#     return (response.choices[0].message["content"] , response["usage"])

# async def aget_completion_from_messages(messages,
#                                             model=LlmModelType,
#                                             temperature=0,
#                                             max_tokens=1000):
#     loop = asyncio.get_event_loop()
#     executor = ThreadPoolExecutor()

#     def run_sync():
#         return openai.ChatCompletion.create(
#             model=model.value,
#             messages=messages,
#             temperature=temperature,
#             max_tokens=max_tokens,
#         )

#     try:
#         response = await loop.run_in_executor(executor, run_sync)
#     except:
#         raise

#     return (response.choices[0].message["content"], response["usage"])
