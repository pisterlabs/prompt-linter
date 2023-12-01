# GPT시인
# ver 0.9

import openai
import streamlit as st

#기본적으로 설정해야하는 부분
openai.api_key = '8a8e7d4d1700468f9bcafc6f48a89216'
openai.api_base = 'https://helloairad.openai.azure.com/'
openai.api_type = 'azure'
openai.api_version = '2023-05-15'

st.header('Welcome to ChatGPT',divider='rainbow')
st.write('안녕하세요 GPT시인 ver 0.9 입니다.')

# 시인의 이름을 입력 받는다. 
name = st.text_input('작가명을 입력해 주세요')
st.write(name + '님 환영합니다.')

# 시의 주제를 입력 받는다. 
subject = st.text_input('시의 주제를 입력하세요')
st.write(subject)

# 시의 구체적인 내용을 입력 받는다. 
content = st.text_area('추가로 하고 싶은 이야기를 입력하세요')

button_click = st.button('시 생성')
if(button_click):

    with st.spinner('Wait for it....'):
        result =openai.ChatCompletion.create(
                    engine='devmodel',
                    messages=[
                        {'role':'system','content':'You are a helpful assistant'},
                        {'role':'user','content':'작가의 이름은 ' + name},
                        {'role':'user','content':'시의 주제는 ' + subject},
                        {'role':'user','content': content},
                        {'role':'user','content':'위의 내용으로 시를 생성해줘'}

                    ]
                )
        
        st.divider()
        st.write('# Result')
        st.write(result.choices[0].message.content)
        st.success('done!')