
import streamlit as st

from langchain.prompts import PromptTemplate 

from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains import LLMChain



load_dotenv()


def main():

    st.set_page_config(page_title="Prompt_Designs",layout="wide")

    #st.title("Prompt_Designs", layout="wide")
    
    
    from langchain.prompts import PromptTemplate 

    from dotenv import load_dotenv

    user_input = st.text_input("Enter any topic")
    

    prompt_template = PromptTemplate(
        input_variables =["user_input"],
        template = "List three facts about {user_input}"
    )



    #formatted = prompt_template.format(topic="elephant")

    #print(formatted)


    openai_llm = OpenAI(verbose=True, temperature=0.1)

    chain =  LLMChain(llm=openai_llm, prompt=prompt_template)

    st.write("Prompt designed in such a way: template = List three facts about {user_input}")

    st.subheader("Configuration Prompt template")
    st.write("Prompt template = How to configure {user_input} provide step by step configuration")

    st.write(f"Prompt template = How to configure {user_input} provide step by step configuration")
    
    st.subheader("Troubleshooting Prompt template")

    st.write(f"Prompt template=what are the the cli show commands for {user_input}")

    if st.button("submit"):

        response = chain.run(user_input)
        st.write(response)

        

    #print(f"response ={response}")


if __name__ == "__main__":
    main()