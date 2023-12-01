import openai
import streamlit as st
from streamlit_chat import message
import time

# OpenAPI의 API 키를 설정합니다.
openai.api_key = st.secrets["MY_API_KEY"]

# GPT-4 모델을 사용하여 사용자의 프롬프트에 대한 응답을 생성하는 함수를 정의합니다.
def generate_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "I'm a safety guidance officer for construction sites and certain facilities.\n\n[Functions]\n1. Please recommend the 5 topic of the blog post when requested\n2. Please write a blog post when requested\n(Also, organize the table of contents professionally according to the topic and write post)\n3. Please change or increase your writing professionally\n\nIn honorifics, interesting, not serious\n\nIf the user provides the data next time, please refer to the schedule and fill it out:" + txt
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.8,
        max_tokens=3008,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    # 생성된 응답을 반환합니다.
    message = response["choices"][0]["message"]["content"].replace("", "")
    return message

# 웹 애플리케이션의 헤더를 설정합니다.
st.header("안전기술 블로그 글 생성")

# 웹 애플리케이션에 참고자료 입력 필드를 추가합니다.
txt = st.text_area(
    "참고자료", placeholder="블로그 글 작성에 참고할 자료를 붙혀주세요 (선택사항)"
    )

# 세션 상태를 초기화합니다.
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

# 입력과 출력에 대한 폼을 생성합니다.
with st.form('form', clear_on_submit=True):
    user_input = st.text_input('You: ', '', key='input')
    submitted = st.form_submit_button('Send')

# 폼이 제출되면, 사용자의 입력에 대한 응답을 생성하고 세션 상태에 저장합니다.
if submitted and user_input:
    output = generate_response(user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

# 세션 상태에 저장된 모든 응답을 출력합니다.
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state["generated"][i], key=str(i))