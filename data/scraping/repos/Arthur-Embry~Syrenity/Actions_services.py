from fastapi.responses import StreamingResponse
from typing import Union, List
from pydantic import BaseModel
import openai
import os
import backoff

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


#set env variable of api key
openai.api_key = os.environ["OPENAI_API_KEY"]


class connect_live_agent_params(BaseModel):
    pass

def connect_live_agent_run(params: connect_live_agent_params):
    """
    ## Description
    swaps the chatbot for a live agent
    """
    #TODO: add code
    return "hello world"

class response_params(BaseModel):
    chat_log: Union[List[str],None]=[""]
    current_user: Union[str,None]="User 1:"
    engine: Union[str,None]="text-davinci-003",
    temperature: Union[float,None] = 0.9
    max_tokens: Union[int,None] = 250
    top_p: Union[float,None] = 1
    frequency_penalty: Union[float,None] = 0
    presence_penalty: Union[float,None] = 0

def response_run(params: response_params):
    """
    ## Description
    responds to a message with GPT response
    """
    prompt = ""
    for i in params.chat_log:
        prompt += i+"\n"
    gpt_iter = completions_with_backoff(
        model="text-davinci-003",
        prompt="""The following is a conversation with an AI diary named Syrenity, whos goal is to listen and clarify what the human is telling you. The diary cares about the human and their health, and is friendly, helpful, accurate, sincere, and can show emotion as their friend.
        \n"""+prompt+"Syrenity: ",
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stop=["User 1:", "User 2:", "Syrenity:"],
        stream=True
    )
    def iterfile():
        for i in gpt_iter:
            yield str(i.choices[0].text)

    return StreamingResponse(iterfile(), media_type="text/plain")


class guide_message_params(BaseModel):
    guidance: Union[str,None]=""
    chat_log: Union[List[str],None]=[""]
    current_user: Union[str,None]="User 1:"
    engine: Union[str,None]="text-davinci-003",
    temperature: Union[float,None] = 0.9
    max_tokens: Union[int,None] = 250
    top_p: Union[float,None] = 1
    frequency_penalty: Union[float,None] = 0
    presence_penalty: Union[float,None] = 0

def guide_message_run(params: guide_message_params):
    """
    ## Description
    responds to a message with GPT response
    """
    prompt = ""
    for i in params.chat_log:
        prompt += i+"\n"
    gpt_iter = completions_with_backoff(
        model=params.engine,
        prompt="""
        The following is a conversation with an AI diary named Syrenity, whos goal is to listen and clarify what the human is telling you. The diary cares about the human and their health, and is friendly, helpful, accurate, sincere, and can show emotion as their friend.
        \n"""+prompt+"\nresponse subject: "+params.guidance+"\n"+"Syrenity: ",
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stop=["User 1:", "User 2:", "Syrenity:"],
        stream=True
    )
    def iterfile():
        for i in gpt_iter:
            yield str(i.choices[0].text)

    return StreamingResponse(iterfile(), media_type="text/plain")

class cache_data_params(BaseModel):
    chat_history: Union[List[str],None]=""
    input: Union[str,None]=""

def cache_data_run(params: cache_data_params):
    """
    ## Description
    caches the data for future use
    """
    #TODO: add code
    return "hello world"

class extract_info_params(BaseModel):
    context: Union[str,None]="",
    engine: Union[str,None]="text-davinci-003",
    temperature: Union[float,None] = 0.9
    max_tokens: Union[int,None] = 250
    top_p: Union[float,None] = 1
    frequency_penalty: Union[float,None] = 0
    presence_penalty: Union[float,None] = 0

def extract_info_run(params: extract_info_params):
    """
    ## Description
    extracts information from the user
    """
    gpt_iter = completions_with_backoff(
        model=params.engine,
        prompt=params.context+"\nGiven the above, extract a few key notes relevant to a therapist, and use dashes for each note:",
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stream=True
    )
    def iterfile():
        for i in gpt_iter:
            yield str(i.choices[0].text)
    return StreamingResponse(iterfile(), media_type="text/plain")
    

class biased_extract_params(BaseModel):
    bias: Union[str,None]="",
    context: Union[str,None]="",
    engine: Union[str,None]="text-davinci-003",
    temperature: Union[float,None] = 0.9
    max_tokens: Union[int,None] = 250
    top_p: Union[float,None] = 1
    frequency_penalty: Union[float,None] = 0
    presence_penalty: Union[float,None] = 0

def biased_extract_run(params: extract_info_params):
    """
    ## Description
    extracts information from the user
    """
    """prompt=params.context+"\nGiven the above, extract \""+params.bias+"\" information and note it with a - in front:","""
    gpt_iter = completions_with_backoff(
        model=params.engine,
        prompt=params.context
        +"\nGiven the above, and the template\n-property:value \n-property:value \n-property:value \netc.\nextract \""
        +params.bias+"\" information:\n-",
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stream=True
    )
    def iterfile():
        yield "-"
        for i in gpt_iter:
            yield str(i.choices[0].text)
    return StreamingResponse(iterfile(), media_type="text/plain")

class guided_message_khi_params(BaseModel):
    context: Union[str,None]="",
    user_data: Union[str,None]="",
    KHI: Union[str,None]="",
    engine: Union[str,None]="text-davinci-003",
    temperature: Union[float,None] = 0.9
    max_tokens: Union[int,None] = 250
    top_p: Union[float,None] = 1
    frequency_penalty: Union[float,None] = 0
    presence_penalty: Union[float,None] = 0

def guided_message_khi_run(params: guided_message_khi_params):
    """
    ## Description
    extracts information from the user
    """


    '''Given this KHI [], and [previous conversation], what is a good question to continue a conversation with the human?'''
    backstory="You are an AI diary named Syrenity, and your goal is to listen and clarify what the human is telling you.  You always care about the human and their health, and are friendly, helpful, accurate, sincere, and can show emotion as their friend."
    gpt_iter = completions_with_backoff(
        model=params.engine,
        
        prompt="Backstory:\n"+backstory
        +"\n\nGiven this list of things we want to know about the human: \""+
        params.KHI+"\", and the conversation: \n\n\""+params.context
        +"\"\n\nas well as the extracted information:\n\n\""+params.user_data
        +"\"\n\nWhat is a good question about N/A information?\n\nA good question would be \"",
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stop=["\""],
        stream=True
    )
    def iterfile():
        for i in gpt_iter:
            yield str(i.choices[0].text)
    return StreamingResponse(iterfile(), media_type="text/plain")


class health_analysis_params(BaseModel):
    context: Union[str,None]="",
    sensitive_extraction: Union[str,None]="mental disorder",
    engine: Union[str,None]="text-davinci-003",
    temperature: Union[float,None] = 0.9
    max_tokens: Union[int,None] = 250
    top_p: Union[float,None] = 1
    frequency_penalty: Union[float,None] = 0
    presence_penalty: Union[float,None] = 0

def health_analysis_run(params: health_analysis_params):
    """
    ## Description
    extracts information from the user
    Don't add to conversation, therapist access only
    """ 
    gpt_iter = completions_with_backoff(
        model=params.engine,
        prompt=params.context+"\nGiven the above, extract \""+params.sensitive_extraction+"\" information and note it with a - in front:\n-",
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stream=True
    )
    def iterfile():
        yield "-"
        for i in gpt_iter:
            yield str(i.choices[0].text)
    return StreamingResponse(iterfile(), media_type="text/plain")


class end_message_params(BaseModel):
    context: Union[str,None]="",
    engine: Union[str,None]="text-davinci-003",
    temperature: Union[float,None] = 0.9
    max_tokens: Union[int,None] = 250
    top_p: Union[float,None] = 1
    frequency_penalty: Union[float,None] = 0
    presence_penalty: Union[float,None] = 0

def end_message_run(params: end_message_params):
    """
    ## Description
    says goodbye to the user
    """
    gpt_iter = completions_with_backoff(
        model=params.engine,
        prompt=params.context+"\nGiven the above, say goodbye to the human:",
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stream=True
    )
    def iterfile():
        for i in gpt_iter:
            yield str(i.choices[0].text)
    return StreamingResponse(iterfile(), media_type="text/plain")