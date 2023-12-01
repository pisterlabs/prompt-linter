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
                "content": "Write a blog post for the person in charge of safety guidance for construction sites and specific facilities. Please recommend five when requesting a topic for a blog post. If you ask me to write a blog about a specific topic, please write it in Korean as long and detailed and professionally as possible. Write in an interesting way with honorific language. If the user asks to extend the length of the text, please write a long text with good quality, including existing information. If you provide data in the following sentence, please refer to it and write an article including the table of contents:" + txt
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.8,
        max_tokens=3008,
        top_p=1,
        frequency_penalty=0.85,
        presence_penalty=0.85
    )
    # 생성된 응답을 반환합니다.
    message = response["choices"][0]["message"]["content"].replace("", "")
    return message

# 웹 애플리케이션의 헤더를 설정합니다.
st.image("image/bloggpt_safety.png")
st.header("안전기술 블로그 GPT")

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
    with st.spinner("Thinking..."):
        output = generate_response(user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

# 세션 상태에 저장된 모든 응답을 출력합니다.
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state["generated"][i], key=str(i))