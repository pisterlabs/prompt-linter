import streamlit as st 
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI


st.title('🦜🔗 PromptPRO')
api_key = st.text_input('Enter your OpenAI Key!')
if api_key:
  try:
    llm = OpenAI(openai_api_key=api_key)

    prompt = PromptTemplate(input_variables=['question'], template="""
        Question: {question}

        Answer: Let's think step by step.
        """)

    llm_chain = LLMChain(prompt=prompt, llm=llm)


    st.info('This is using the ggml-gpt4all-j-v1.3-groovy model!')
    prompt = st.text_input('Enter your prompt here!')
    if prompt: 
        response = llm_chain.run(prompt)
        st.write(response)
        st.write('Finished')
  except:
    st.write("Seems like there's an issue with your api key")
