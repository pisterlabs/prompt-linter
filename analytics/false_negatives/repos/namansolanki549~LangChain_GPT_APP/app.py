import streamlit as st 
from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All

# PATH = '/Users/namansolanki/tensorflow-test/Projects/DataScience/GPT4ALL/ggml-gpt4all-l13b-snoozy.bin'
# PATH = '/Users/namansolanki/tensorflow-test/Projects/DataScience/GPT4ALL/ggml-gpt4all-j-v1.3-groovy.bin'
PATH = '/Users/namansolanki/tensorflow-test/Projects/DataScience/GPT4ALL/ggml-gpt4all-j-v1.3-groovy.bin'
llm = GPT4All(model=PATH, verbose=True)

prompt = PromptTemplate(input_variables=['question'], template="""
    Question: {question}
    
    Answer: Let's think step by step.
    """)

llm_chain = LLMChain(prompt=prompt, llm=llm)

st.title('🦜🔗 PromptPRO')
st.info('This is using the ggml-gpt4all-j-v1.3-groovy model!')
prompt = st.text_input('Enter your prompt here!')
if prompt: 
    response = llm_chain.run(prompt)
    print(response)
    st.write(response)
    st.write('Finished')
