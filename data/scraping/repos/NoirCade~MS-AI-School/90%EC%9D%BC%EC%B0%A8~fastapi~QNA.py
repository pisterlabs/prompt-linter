import os
import openai
import streamlit as st
import requests
import json
openai.api_key = ("")
REST_API_KEY = ''


def kogpt_api(prompt, max_tokens=1, temperature=1.0, top_p=1.0, n=1):
  r = requests.post(
    'https://api.kakaobrain.com/v1/inference/kogpt/generation',
    json={
      'prompt': prompt,
      'max_tokens': max_tokens,
      'temperature': temperature,
      'top_p': top_p,
      'n': n
    },
    headers={
      'Authorization': 'KakaoAK ' + REST_API_KEY,
      'Content-Type': 'application/json'
    }
  )
  # 응답 JSON 형식으로 변환
  response = json.loads(r.content)
  return response




def QNA():
  col1, col2,col3 = st.columns(3)
  with col1:
    st.subheader("i'm chatgpt. ask me anything, everything to english")
    name = st.text_area('그래도 gpt라서 이상한거 물어보거나 부정확한 말들을 할 수 있음.')

    response = openai.Completion.create(
      model="text-davinci-003",
      prompt=f"Q:{name} A:",
      temperature=0,
      max_tokens=100,
      top_p=1,
      frequency_penalty=0.0,
      presence_penalty=0.0,
      stop=["\n"]
    )
    response = response['choices']
    response = response[0]
    response = response['text']

    if st.button("ask!"):
      st.write(response)
  with col3:
    st.subheader("i'm kogpt. ask me anything, everything to korean")
    kogptname = st.text_area('부정확하고 오류가 잘남')
    kogptname = kogptname+'\n답:'


    responsekogpt = kogpt_api(prompt=kogptname, max_tokens=250, temperature=0.2, top_p=1.0, n=1)
    responsekogpt = responsekogpt['generations']
    responsekogpt = responsekogpt[0]
    responsekogpt = responsekogpt['text']
    responsekogpt = responsekogpt.split('\n')[0]
    responsekogpt = responsekogpt.split('.')[0]
    responsekogpt = responsekogpt.split('^')[0]
    responsekogpt = responsekogpt.split('▶')[0]
    responsekogpt = responsekogpt.split('/')[0]
    responsekogpt = responsekogpt.split('#')[0]

    if st.button("ask!!"):
      st.write(responsekogpt)

    # 예제 Textfile
  # 예외처리
  # st.write('text')  # df, err, func, keras
