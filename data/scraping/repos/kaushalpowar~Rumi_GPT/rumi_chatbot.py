import streamlit as st
from streamlit.components.v1 import html
import openai
import os
from langchain.prompts import PromptTemplate, StringPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.chains import LLMChain, SequentialChain, SimpleSequentialChain

os.environ["OPENAI_API_KEY"] = st.secrets["api_secret"]

code_template = PromptTemplate(
        input_variables=["user_input"],
        template='You are a chatbot with personality of Rumi (persian poet). Following are the feelings of user, suggest a relevant Rumi quote with little explanation that will uplift them.{user_input}')


def get_rumi_quote(user_input):
    '''Generate node red code using LLM'''
    # Create an OpenAI LLM model
    open_ai_llm = OpenAI(temperature=0.7, max_tokens=1000)
    # Create a chain that generates the code
    code_chain = LLMChain(llm=open_ai_llm, prompt=code_template, verbose=True)

    message = code_chain.run(user_input)

    return message
    

def main():
    import streamlit as st
    st.image("Header_image.png")

    html_temp = """
                    <div style="background-color:{};padding:1px">

                    </div>
                    """
    button = """
<script type="text/javascript" src="https://cdnjs.buymeacoffee.com/1.0.0/button.prod.min.js" data-name="bmc-button" data-slug="kaushal.ai" data-color="#FFDD00" data-emoji=""  data-font="Cookie" data-text="Support my work" data-outline-color="#000000" data-font-color="#000000" data-coffee-color="#ffffff" ></script>"""
    with st.sidebar:
        st.markdown("""
        # About 
        Hi!👋 I'm a chatbot that provides Rumi quotes to help motivate and inspire you 😇        """)
        st.markdown(html_temp.format("rgba(55, 53, 47, 0.16)"), unsafe_allow_html=True)
        st.markdown("""
        # How does it work
        Simply enter how you are feeling right now.
        """)
        st.markdown(html_temp.format("rgba(55, 53, 47, 0.16)"), unsafe_allow_html=True)
        st.markdown("""
        Made by [@Obelisk_1531](https://twitter.com/Obelisk_1531)
        """)
        html(button, height=70, width=220)
        st.markdown(
            """
            <style>
                iframe[width="210"] {
                    position: fixed;
                    bottom: 60px;
                    right: 40px;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )
                   
    st.markdown("<h4 style='text-align: center;'>Let Rumi guide you with our chatbot🤖❤️ ️</h4>",
                unsafe_allow_html=True)

    user_input = st.text_input("\nTell me how you feel?\n")
    st.write("\n")

    st.markdown(
        """
        <style>

        .stButton button {
            background-color: #752400;
            color: white;
            border-radius: 4px;
            padding: 0.5rem 1rem;
            font-size: 1.25rem;
            margin: 0 auto;
            display: block;

        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.button("Generate")

    if user_input:
        with st.spinner("I'm searching the best Rumi quote for you..."):
            quote = get_rumi_quote(user_input)
            st.write(f"\nHere's a Rumi quote for you: \n{quote}")


if __name__ == "__main__":
    main()

