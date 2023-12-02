# 유튜브 동영상을 요약하고 번역하는 웹 앱

import my_yt_tran  # 유튜브 동영상 정보와 자막을 가져오기 위한 모듈 임포트
import my_text_sum # 텍스트를 요약하기 위한 모듈
import streamlit as st
import openai
import os
import tiktoken
import textwrap
import deepl

# 텍스트의 토큰 수를 계산하는 함수(모델: "gpt-3.5-turbo")
def calc_token_num(text, model="gpt-3.5-turbo"):
    enc = tiktoken.encoding_for_model(model)
    encoded_list = enc.encode(text) # 텍스트 인코딩해 인코딩 리스트 생성
    token_num= len(encoded_list)    # 인코딩 리스트의 길이로 토큰 개수 계산
    
    return token_num

# 토큰에 따라 텍스트를 나눠 분할하는 함수
def divide_text(text, token_num):
    req_max_token = 2000 # 응답을 고려해 설정한 최대 요청 토큰
    
    divide_num = int(token_num/req_max_token) + 1 # 나눌 계수를 계산
    divide_char_num = int(len(text) / divide_num) # 나눌 문자 개수 
    divide_width =  divide_char_num + 20 # wrap() 함수로 텍스트 나눌 때 여유분 고려해 20 더함

    divided_text_list = textwrap.wrap(text, width=divide_width)
    
    return divide_num, divided_text_list

# 유튜브 동영상을 요약하는 함수
def summarize_youtube_video(video_url, selected_lang, trans_method):
    
    if selected_lang == '영어':
        lang = 'en' 
    else:
        lang = 'ko' 
    
    # 유튜브 동영상 플레이
    st.video(video_url, format='video/mp4') # st.video(video_url) 도 동일

    # 유튜브 동영상 제목 가져오기
    _, yt_title, _, _, yt_duration = my_yt_tran.get_youtube_video_info(video_url)
    st.write(f"[제목] {yt_title}, [길이(분:초)] {yt_duration}") # 제목 및 상영 시간출력
    
    # 유튜브 동영상 자막 가져오기
    yt_transcript = my_yt_tran.get_transcript_from_youtube(video_url, lang)

    # 자막 텍스트의 토큰 수 계산
    token_num = calc_token_num(yt_transcript)
    
    # 자막 텍스트를 분할해 리스트 생성
    div_num, divided_yt_transcripts = divide_text(yt_transcript, token_num)

    st.write("유튜브 동영상 내용 요약 중입니다. 잠시만 기다려 주세요.") 
    
    # 분할 자막의 요약 생성
    summaries = []
    for divided_yt_transcript in divided_yt_transcripts:
        summary = my_text_sum.summarize_text(divided_yt_transcript, lang) # 텍스트 요약
        summaries.append(summary)
        
    # 분할 자막의 요약을 다시 요약     
    _, final_summary = my_text_sum.summarize_text_final(summaries, lang)

    if selected_lang == '영어':
        shorten_num = 200 
    else:
        shorten_num = 120 
        
    shorten_final_summary = textwrap.shorten(final_summary, shorten_num, placeholder=' [..이하 생략..]')
    st.write("- 자막 요약(축약):", shorten_final_summary) # 최종 요약문 출력 (축약)
    # st.write("- 자막 요약:", final_summary) # 최종 요약문 출력

    if selected_lang == '영어': 
        if trans_method == 'OpenAI':
            trans_result = my_text_sum.traslate_english_to_korean_using_openAI(final_summary)
        elif trans_method == 'DeepL':
            trans_result = my_text_sum.traslate_english_to_korean_using_deepL(final_summary)

        shorten_trans_result = textwrap.shorten(trans_result, 120 ,placeholder=' [..이하 생략..]')
        st.write("- 한국어 요약(축약):", shorten_trans_result) # 한국어 번역문 출력 (축약)
        #st.write("- 한국어 요약:", trans_result) # 한국어 번역문 출력
        
# ------------------- 콜백 함수 --------------------
def button_callback():
    st.session_state['input'] = ""
    
# ------------- 사이드바 화면 구성 --------------------------  
st.sidebar.title("요약 설정 ")
url_text = st.sidebar.text_input("유튜브 동영상 URL을 입력하세요.", key="input")

clicked_for_clear = st.sidebar.button('URL 입력 내용 지우기',  on_click=button_callback)

yt_lang = st.sidebar.radio('유튜브 동영상 언어 선택', ['한국어', '영어'], index=1, horizontal=True)
    
if yt_lang == '영어':
    trans_method = st.sidebar.radio('번역 방법 선택', ['OpenAI', 'DeepL'], index=1, horizontal=True)
else:
    trans_method = ""

clicked_for_sum = st.sidebar.button('동영상 내용 요약')

# ------------- 메인 화면 구성 --------------------------     
st.title("유튜브 동영상 요약")

# 텍스트 입력이 있으면 수행
if url_text and clicked_for_sum: 
    yt_video_url = url_text.strip()
    summarize_youtube_video(yt_video_url, yt_lang, trans_method)
