
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from dotenv import load_dotenv
load_dotenv()
import openai
import configparser
import json
import streamlit as st
openai.api_key=os.getenv("openai_api_key")

# openai_api_key = os.getenv("openai_api_key")

def get_conversation_buffer_memory(conBuffWindow=3):
    return ConversationBufferWindowMemory(k=conBuffWindow,return_messages=True)



def get_response(model, query, context):

    # prompt_template, temperature, prompt = get_gpt_config(source)
    prompt="""\n\nInstructions for Response Generation:\n\n
            Kindly exhibit empathetic responses to greeting messages and provide thoughtful  replies.
            The context provided is from "Transcription" which is a transcription of some youtube video.
            Always refer "Transcription" as the context in the response instead of saying "context".
            Format the response as a valid json with keys "Bot Response" and "Relevancy".Format the "Relevancy" value as a boolean based on whether the Query is based on "Transcription" provided in the user_input or not.
            If the query answer is available in  "Transcription"  for answering, set "Relevancy" as true else set it as false.\n
            "Bot Response" is a multiline json string, make sure that the format it as a valid json string.
            For general conversational interactions or inquiries not requiring specialist knowledge, feel free to respond naturally and appropriately without relying on the "Transcription".
            Make sure you always format the response as  Json as mentioned above and keep your answer within 50 words. 
            """
    
    system_role = prompt
   


    # response_text = "{\"Relevancy\":false,\"Bot Response\": \"The information provided in the ABCD Knowedge Library does not contain any relevant information about washing with hope.\"}"
    user_input = f"\n\nQuery: {query}\n\n Transcript:\n {context}\n\n "



    # response_text_2 = "{\"Relevancy\":true,\"Bot Response\": \"Washing with hope is a culture in florida\"}"
    # user_input_2 = f"\n\nQuery: {query}\n\n T:\n {context}\n\n "
   
    response_list = [
        {"role": "system", "content": system_role},
        # {"role": "user", "content": example_text},
        # {"role": "assistant", "content": response_text},
        {"role": "user", "content": user_input},
        # {"role": "user", "content": example_text_2},
        # {"role": "assistant", "content": response_text_2},
        # {"role": "user", "content": user_input_2},
    ]


    response = openai.ChatCompletion.create(
        model=model,
        temperature=0,
        messages=response_list
    )
   
    gpt_response = response["choices"][0]["message"]["content"]

    print("user Query-Input:")
    print(user_input)

    print("bot Output:")
    print(gpt_response)
    st.write(gpt_response)

    # try:
    response_dict = json.loads(gpt_response)

    response = response_dict["Bot Response"]
    within_knowledge_base = response_dict["Relevancy"]

    print("*" * 50)
    print("is query related to knowledge base: ", within_knowledge_base)
    print("*" * 50)
    try:
        response= json.loads(response)
        response = response["Bot Response"]
    except:
        pass
    return response




def normal_talk(model, query, context):
    prompt="""\n\nInstructions for Response Generation:\n\n
            Kindly exhibit empathetic responses to greeting messages and provide thoughtful  replies.
            The context provided is from "Transcription" which is a transcription of some youtube video.
            Always refer "Transcription" as the context in the response instead of saying "context".
            Format the response as a valid json with keys "Bot Response" and "Relevancy".Format the "Relevancy" value as a boolean based on whether the Query is based on "Transcription" provided in the user_input or not.
            If the query answer is available in  "Transcription"  for answering, set "Relevancy" as true else set it as false.\n
            "Bot Response" is a multiline json string, make sure that the format it as a valid json string.
            For general conversational interactions or inquiries not requiring specialist knowledge, feel free to respond naturally and appropriately without relying on the "Transcription".
            Make sure you always format the response as  Json as mentioned above and keep your answer within 50 words. 
            """
    system_role = prompt
    buffer_memory = get_conversation_buffer_memory()
    example_text = f"""\n\nCONVERSATION LOG: \n
                    Human: hi
                    Bot Response: hi
                    Query: What is the meaning or significance behind Washing with hope?
                    Transcript: I live in florida.
                    """

    response_text = "{\"Relevancy\":false,\"Bot Response\": \"The information provided in the transcript does not contain any relevant information about washing with hope.\"}"
    user_input = f"\n\nCONVERSATION LOG: \n{buffer_memory}\n\nQuery: {query}\n\n  Transcript:\n {context}\n\n "


    

    response_list = [
        {"role": "system", "content": system_role},
        {"role": "user", "content": example_text},
        {"role": "assistant", "content": response_text},
        {"role": "user", "content": user_input}
    ]
    response = openai.ChatCompletion.create(
        model=model,
        temperature=0,
        messages=response_list
    )
   


    gpt_response = response["choices"][0]["message"]["content"]

    print("user Query-Input:")
    print(user_input)

    print("bot Output:")
    print(gpt_response)
    st.write(gpt_response)

    # try:
    response_dict = json.loads(gpt_response)

    response = response_dict["Bot Response"]
    within_knowledge_base = response_dict["Relevancy"]

    print("*" * 50)
    print("is query related to knowledge base: ", within_knowledge_base)
    print("*" * 50)
    try:
        response= json.loads(response)
        response = response["Bot Response"]
    except:
        pass
    return response

# if __name__=="__main__":
    # normal_talk(model="gpt-3.5-turbo-16k", query="What is the meaning or significance behind Washing with hope?", context="I live in florida.")
    