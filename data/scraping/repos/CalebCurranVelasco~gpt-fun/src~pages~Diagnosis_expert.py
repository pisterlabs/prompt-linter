from dotenv import find_dotenv, load_dotenv
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents.load_tools import get_all_tool_names
from langchain import ConversationChain
from langchain.chat_models import ChatOpenAI
import streamlit as st
from htmlTemplates import css, bot_template, user_template

def main():
    load_dotenv(find_dotenv())
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    st.set_page_config(page_title="Medical Diagnosis", page_icon=":dna:")
    st.write(css, unsafe_allow_html=True)

    llm = OpenAI(model_name="gpt-3.5-turbo")

    # symptomes = input("Please enter you symptomes: ")
    st.header("	:stethoscope: Medical Diagnosis :dna:")
    symptomes = st.text_input("Please enter your symptomes:")

    if symptomes:


        prompt = PromptTemplate(
            input_variables=["symptomes"],
            template="Given the following symptomes: {symptomes}. Give a diagnosise for the patient.",
        )

        chain = LLMChain(llm=llm, prompt=prompt)
        # print(chain.run(symptomes))
        st.write(chain.run(symptomes))

        # prompt.format(product=symptomes)
        # print(llm(prompt))

        # prompt = "Write a poem about python and ai"
        # print(llm(prompt))
    with st.sidebar:
        st.markdown(
            "## How to use\n"
            "1. Input any symptomes into the text box\n"  
            "2. Press enter so that the model can process them\n"
            "3. The model will return a potential diagnosis."
        )
        st.markdown("---")

        st.subheader("Please remember that this uses an AI language model, and the responses are based on predictions. It's crucial to consult with your doctor or a qualified healthcare professional for accurate and personalized medical advice.")

if __name__=='__main__':
    main()