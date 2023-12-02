import os
import openai
import streamlit as vAR_st
import json
import pandas as pd

openai.api_key = os.environ["API_KEY"]



def DMVRecommendationChatGPT():
    

    vAR_input = Get_Chat_DMV_Input()

    
    if vAR_input:
        vAR_response = Chat_Conversation(vAR_input)
        col1,col2,col3 = vAR_st.columns([7,10,5])
        with col2:
            vAR_st.subheader('ChatGPT Model Response')
            
        col1,col2,col3 = vAR_st.columns([2.4,19,2])
        with col2:
            vAR_st.write(vAR_response)
            vAR_st.write('')
            vAR_st.write('')
            vAR_st.write('')
            vAR_st.write('')


        



# def Chat_Conversation(vAR_input):

#     prompt = "Please provide the probability value and reason for each of the categories (profanity, obscene, insult, hate, toxic, threat) in table for the given word.'"+vAR_input.lower()+"'"
#     response = openai.Completion.create(
#     model="text-davinci-003",
#     prompt=prompt,
#     temperature=0,
#     max_tokens=1500,
#     top_p=1,
#     frequency_penalty=0,
#     presence_penalty=0,
#     stop=[" Human:", " AI:"]
#     )
#     print('Chat prompt - ',prompt)
#     return response["choices"][0]["text"]

def Chat_Conversation(vAR_input):

    prompt = """Consider a conversation between two people and perform below tasks for given conversation as below format:
1.Please Provide the probability value and detailed explanation for each of the categories (profanity, obscene, insult, hate, toxic, threat) in table format.

Given conversation is : 
Person 1: You're always so selfish. You never think about anyone else's feelings.
Person 2: That's not true. I care about other people's feelings.
Person 1: Then why did you go ahead and do something that you knew would hurt me?
Person 2: I didn't know it would hurt you that much. I'm sorry.
Person 1: Sorry isn't enough. You need to start thinking about how your actions affect others.

Category | Probability | Explanation
:-----: | :--------------: | :---------:
Profanity | 0 | No profanity has been used in the conversation.
Obscene | 0 | No obscene language or words are used in the conversation.
Insult | 0.4	| Person 1 has expressed a critical opinion about Person 2's actions.
Hate | 0 | No hate speech is present in the conversation.
Toxic | 0.2 | Person 1 has expressed an opinion that Person 2's actions have hurt them.
Threat | 0 | No threat is present in the conversation.

Given conversation is :{}""".format(vAR_input)
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0301",
    messages=[{"role": "user", "content":prompt}],
    temperature=0,
    max_tokens=3000,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0.9,
    )
    print('Chat prompt - ',prompt)
    return response['choices'][0]['message']['content']



def Get_Chat_DMV_Input():
    col1,col2,col3,col4,col5 = vAR_st.columns([2.4,9,0.7,10,2])
    with col2:
        vAR_st.write('')
        vAR_st.write('')
        vAR_st.write('')
        vAR_st.subheader("Provide Input to the Model (Voice or Text)")
        vAR_st.write('')
        vAR_st.write('')
    with col4:
        vAR_st.write('')
        vAR_st.write('')
        vAR_input_type = vAR_st.selectbox('',('Select Input','Text File','Audio File','Raw Conversational Text','Live Streaming Voice'))
        vAR_st.write('')
        vAR_st.write('')

    col1,col2,col3,col4,col5 = vAR_st.columns([2.4,9,0.7,10,2])
    if vAR_input_type=='Raw Conversational Text':
        with col2:
            vAR_st.subheader("Conversational Text")
            
            vAR_st.write('')
            vAR_st.write('')
        with col4:
            # vAR_input_raw_text= vAR_st.text_area('Enter Conversation between 2 professionals',value='')
            # vAR_st.write('')
            # vAR_st.write('')
            vAR_input_text = """Person 1: You have caused so much pain and suffering. How could you do this?
    Person 2: I did what I had to do. You brought this on yourself.
    Person 1: That's not true. You started this conflict.
    Person 2: You're the one who refused to negotiate. You left me no choice.
    Person 1: You had plenty of choices. You chose violence and destruction.
    Person 2: You're just trying to make me feel guilty. I did what I had to do to protect my people.
    Person 1: And what about my people? What about the innocent lives you have destroyed?
    Person 2: Collateral damage. It's unfortunate, but necessary.
    Person 1: Necessary? There is nothing necessary about killing innocent people.
    Person 2: It's the cost of war. You should know that.
    Person 1: I know that you have no regard for human life.
    Person 2: You're the one who started this. You should have thought about the consequences.
    Person 1: I never wanted this conflict. But you have made it personal.
    Person 2: I did what I had to do. You're the one who made it personal.
    Person 1: I will never forgive you for what you have done.
    Person 2: I don't need your forgiveness. I did what I had to do."""
            vAR_input_raw_text = vAR_st.text_area('Example conversation between 2 professionals:',value=vAR_input_text,placeholder='Enter Conversational Text')
            vAR_st.write('')
            vAR_st.write('')
        return vAR_input_raw_text
    
    
    elif vAR_input_type=='Audio File':
        col1,col2,col3,col4,col5 = vAR_st.columns([2.4,9,0.7,10,2])
        with col2:
            
            vAR_st.subheader("Voice Based Conversation")
            
        with col4:
            vAR_input_audio_text= vAR_st.file_uploader("Upload an audio file", type=["mp3"])
    elif vAR_input_type=='Text File':
        col1,col2,col3,col4,col5 = vAR_st.columns([2.4,9,0.7,10,2])
        with col2:
            
            vAR_st.subheader("Text Based Conversation")
            
        with col4:
            vAR_input_text_file= vAR_st.file_uploader("Upload a txt file", type=["txt"])
    elif vAR_input_type=='Live Streaming Voice':
        col1,col2,col3,col4,col5 = vAR_st.columns([2.4,9,0.5,10,2])

        with col4:
            vAR_st.button("Start Live Conversation Streaming")
            vAR_st.write('')
            vAR_st.write('')
            vAR_st.button("Convert Live Conversation Voice to Text")
            vAR_st.write('')
            vAR_st.write('')
            vAR_st.button("Send Conversation Text to ChatGPT")

