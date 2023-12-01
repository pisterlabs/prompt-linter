import os
from apikey import apikey, serpapikey
import streamlit as st
from langchain.llms import OpenAI
from langchain import PromptTemplate

os.environ['OPENAI_API_KEY'] = apikey

# PROMPT
st.title('AMUBHYA AI | EXPERIMENTAL')
user_input = st.text_input('Whatever You Want !')


llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo",openai_api_key= apikey)

json_example= """ 
            {
                "title": "",
                "description": "",
                "materials" : [
                    {
                        "name": "",
                        "description": "",
                    }
                ]
            } 
        """
prompt_template = """
         I want to know {text} self-learning material and the points 
        for each material are as complete as possible in indonesian language and the difficulty level of beginer in parsable JSON format : \n
        {json_example}
                  """

prompt = PromptTemplate(
    input_variables=['text','json_example'],
    template=prompt_template
)

if user_input:
    final_prompt = prompt.format(text=user_input,json_example=json_example)
    st.write(final_prompt)

    response = llm(final_prompt)
    st.json(response)