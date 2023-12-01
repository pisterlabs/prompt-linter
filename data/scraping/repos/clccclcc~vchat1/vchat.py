#import streamlit as st
#import openai

import streamlit as st
# audiorecorder 패키지 추가
from audiorecorder import audiorecorder
# OpenAI 패키기 추가
import openai
# 파일 삭제를 위한 패키지 추가
import os
# 시간 정보를 위핸 패키지 추가
from datetime import datetime
# 오디오 array 비교를 위한 numpy 패키지 추가
import numpy as np
# TTS 패키기 추가
from gtts import gTTS
# 음원파일 재생을 위한 패키지 추가
import base64



# openai.api_key = "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"
# openai.api_key = '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' # 990911

def main_page():
    st.set_page_config(page_title='음성비서 프로그램', layout="wide")
    st.header("음성 비서")
    
    st.markdown("---")
    
    with st.expander("About",expanded=True):
        st.write(
    """
    - by clcc  2023.9.11.
    - 음성->text-> GPT -> text -> 음성 구조
    - Have a nice day!
    
    """
        )
        
        st.markdown("")
        
def left_page():
    with st.sidebar:
        openai.api_key=st.text_input(label="api key", placeholder="api key 입력", value="" , type="password")
        
        st.markdown("---")
        
        # model = st.radio(label="model", options=["gpt-3.5-turbo","gpt-4"])
        st.session_state['model'] = st.radio(label="model", options=["gpt-3.5-turbo","gpt-4"])
        
        st.markdown("---")
            
        if st.button(label="초기화"):
            print("초기화 button pressed")
            ini_session()
        
def ini_session():
    if "chat" not in st.session_state:
        st.session_state['chat'] = []
        
    if 'message'not in st.session_state:
        st.session_state['messages'] = [{'role':'system',
                                        'content':'You are a thoughtful assistant. Response to al input in 25 words and answer in korea.'}]
        
    if 'check_audio' not in st.session_state:
        st.session_state['check_audio'] = []
        
    if 'model' not in st.session_state:
        st.session_state['model'] = 'gpt-3.5-turbo'
        
    if 'flag_start' not in st.session_state:
        st.session_state['flag_start'] = False
        

# from audiorecorder import audiorecorder
# import numpy as np   

def STT(audio):
    # 파일 저장
    # index = np.random.randint(0,10)
    # filename=f'input{index}.mp3'
    
    filename = 'input.mp3'

    audio.export(filename)    
        
    # 음원 파일 열기
    audio_file = open(filename, "rb")
    #Whisper 모델을 활용해 텍스트 얻기
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    audio_file.close()
    # 파일 삭제
    os.remove(filename)
    return transcript["text"]

def myrecode():
    st.subheader('질문하기')
    
    
    # 음성 녹음하기
    audio = audiorecorder("클릭하여 녹음하기", "녹음 마치기")
    
    if len(audio) > 0 and not np.array_equal(audio,st.session_state["check_audio"]):
        # 음성 재생 
        # m1
        # st.audio(audio.tobytes())   # error 
        
        # m2 
        audio.export('audio.mp3')
        with open('audio.mp3','rb') as f:
            data = f.read()
        st.audio(data)      
        
        # 음원 파일에서 텍스트 추출
        question = STT(audio)     
        # print('text:',question)  
        
        
        # 채팅을 시각화하기 위해 질문 내용 저장
        now = datetime.now().strftime("%H:%M")
        st.session_state["chat"] = st.session_state["chat"]+ [("user",now, question)]
        
        # GPT 모델에 넣을 프롬프트를 위해 질문 내용 저장
        st.session_state["messages"] = st.session_state["messages"]+ [{"role": "user", "content": question}]
        
        # audio 버퍼 확인을 위해 현 시점 오디오 정보 저장
        st.session_state["check_audio"] = audio
        st.session_state['flag_start'] =True
        
        # st.markdown("---")
        # st.subheader("입력한 내용:")
        st.write(question)
        # st.markdown("---")
        # st.write(st.session_state["chat"])
        # st.write(st.session_state["messages"])

def check_bill_ex(response, wrate=1323.43,prompt_rate=0.0015, completion_rate=0.002):
    prompt_tokens = response['usage']["prompt_tokens"]
    completion_tokens= response['usage']["completion_tokens"]
    
    total_bill = prompt_tokens * prompt_rate/1000 + completion_tokens * completion_rate/1000
    total_won = total_bill * wrate
       
    return total_won, total_bill ,prompt_tokens, completion_tokens
def ask_gpt(prompt, model):
    response = openai.ChatCompletion.create(model=model, messages=prompt)
    system_message = response["choices"][0]["message"]
    return system_message["content"],response 

def TTS(response):
    # gTTS 를 활용하여 음성 파일 생성    
    # index = np.random.randint(0,10)
    # filename = f"output{index}.mp3"
    filename = 'optput.mp3'
    
    tts = gTTS(text=response,lang="ko")
    tts.save(filename)

    # 음원 파일 자동 재성
    with open(filename, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
    
    
        md = f"""
            <audio autoplay="True">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(md,unsafe_allow_html=True,)
    # 파일 삭제
    os.remove(filename)
        
def myresponse():
    st.subheader("답변 듣기")
    
    if st.session_state['flag_start']:
        #ChatGPT에게 답변 얻기    
        response ,res = ask_gpt(st.session_state["messages"], st.session_state['model'])
        # st.write(response)
        st.session_state['flag_start'] = False 
        
        
        won,bil,token_in,token_out = check_bill_ex(res)
        st.write(f' {won}won ,{bil} $ , {token_in} token_in ,{token_out} token_out')
        # st.write(f' {token_in} token_in ,{token_out} token_out')
        st.markdown("---")
        
        # GPT 모델에 넣을 프롬프트를 위해 답변 내용 저장
        st.session_state["messages"] = st.session_state["messages"]+ [{"role": "system", "content": response}]

        # 채팅 시각화를 위한 답변 내용 저장
        now = datetime.now().strftime("%H:%M")
        st.session_state["chat"] = st.session_state["chat"]+ [("bot",now, response)]

        # 채팅 형식으로 시각화 하기
        for sender, time, message in st.session_state["chat"]:
            if sender == "bot": # bot
                st.write(f'<div style="display:flex;align-items:center;"><div style="background-color:#007AFF;color:white;border-radius:12px;padding:8px 12px;margin-right:8px;">{message}</div><div style="font-size:0.8rem;color:gray;">{time}</div></div>', unsafe_allow_html=True)
                st.write("")
            else: # bot 
                print('sender:',sender)
                st.write(f'<div style="display:flex;align-items:center;justify-content:flex-end;"><div style="background-color:lightgray;border-radius:12px;padding:8px 12px;margin-left:8px;">{message}</div><div style="font-size:0.8rem;color:gray;">{time}</div></div>', unsafe_allow_html=True)
                st.write("")
        # gTTS 를 활용하여 음성 파일 생성 및 재생
        TTS(response)
                            
            
def main():
    # 음성비서 프로그램 
    ini_session()
    
    main_page()
    
    left_page()
    
    col1,col2 = st.columns(2)
    with col1:        
        myrecode()
        
    with col2:
        myresponse()
    

main()