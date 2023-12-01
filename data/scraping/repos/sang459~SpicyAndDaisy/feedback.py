# 피드백 페이지 (feedback)

import streamlit as st
import openai
import json
import re
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(initial_sidebar_state="collapsed")

st.markdown('버그 제보 : https://open.kakao.com/o/sr6Mcjxf')

st.markdown("""
            <style>
            [data-testid="stSidebar"] {
                display: none
            }

            [data-testid="collapsedControl"] {
                display: none
            }
            </style>
            """, unsafe_allow_html=True)

try:
    username = st.session_state['username']
except Exception as e:
    print(e)
    switch_page('main')
    
f"{username}님께 SPICY가 할 말이 있다네요..."

with open('users.json', 'r', encoding='utf-8') as file:
    config = json.load(file)

OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
RAPID_API_KEY = st.secrets['RAPID_API_KEY']

# user의 page 정보 갱신 및 저장
with open('users.json', 'w', encoding='utf-8') as file:
    config[username]['page'] = 'feedback'
    json.dump(config, file, ensure_ascii=False)


def starts_with_hangul(text):
    hangul = re.compile('^[가-힣]')
    result = hangul.match(text)

    return result is not None


st.image('sources/hujup.jpg')
# st.info('_SPICY says..._\n\n' + st.session_state['feedback'])
# st.info('_SPICY says..._\n\n' + st.session_state['translated_response'])

saved_feedback = config[username]['feedback']
if starts_with_hangul(saved_feedback): # 유저가 번역 후 재접속했다면
    fin_res = saved_feedback
    st.info(saved_feedback)
else:
    # streaming
    fin_res = ''
    res_box = st.empty()
    report = []
    result = ''

    kor_response = openai.ChatCompletion.create(
        model = "gpt-4",
        messages = [{"role": "system", "content": "당신은 전문 번역가입니다. 다음 대사를 한글로 번역하세요. 존댓말을 사용하세요. 'you'는 '당신'으로 번역하세요.: " + st.session_state['feedback']}],
        stream=True,
        temperature=0.14,
        top_p=0.18
        )
    
    for resp in kor_response:
        try:
            report.append(resp['choices'][0]['delta']['content'])
        except KeyError:
            report.append(' ')
        result = "".join(report)
        res_box.info('_SPICY says..._\n\n' + result)

    # chat_history.append({"role": "assistant", "content": result}) (히스토리 추가시...)
    fin_res += result
    res_box.info('_SPICY says..._\n\n' + result)

with open('users.json', 'w', encoding='utf-8') as file:
    config[username]['feedback'] = fin_res
    json.dump(config, file, ensure_ascii=False)

if st.button('내일 목표 설정하러 가기'):
    switch_page('set_goal')