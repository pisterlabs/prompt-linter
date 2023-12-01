import os
import openai
import streamlit as st
openai.api_key = ("")


def bugfixer():

  st.subheader("T.T i can't solve error")
  name = st.text_area('dont worry, chatGPT can help you! be happy!:D')


  response = openai.Completion.create(
    model="code-davinci-002",
    prompt=f"##### Fix bugs in the below function\n### Buggy Python \n{name} \n ### Fixed Python",
    temperature=0,
    max_tokens=182,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stop=["###"]
  )
  response = response['choices']
  response = response[0]
  response = response['text']

  if st.button("find!"):
    st.write(response)

    # 예제 Textfile
  # 예외처리
  # st.write('text')  # df, err, func, keras
