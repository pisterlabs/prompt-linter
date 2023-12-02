import streamlit as st
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import webbrowser
# Set page config
st.set_page_config(
    page_title="What's TRUE: Financial Advisor",
    page_icon="💼",
    layout="wide",
)

# Custom CSS to style the app
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://example.com/finance_background_image.jpg');  /* Replace with your image URL */
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .title-text {
        font-size: 36px;
        color: #ffffff;  /* White text color */
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App title and description
if st.button("Go to Smallcut"):
    # URL to redirect to
    redirect_url = "http://localhost:4242/payment"
    link = f'<a href="{redirect_url}" target="_blank">Go to Smallcut</a>'
    st.markdown(link, unsafe_allow_html=True)
st.title("🌟 What's TRUE: Your Financial Advisor")
st.markdown("Aladin + Us is the best solution you've got! 💰",
            unsafe_allow_html=True)

API = "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"
user_question = st.text_input(
    "Please enter your age, financial goals, short term and long term goal", value="")

llm = OpenAI(temperature=0.7, openai_api_key=API)

if st.button("Tell me about it", key="tell_button"):
    template = "{question}\n\n"
    prompt_template = PromptTemplate(
        input_variables=["question"], template=template)
    question_chain = LLMChain(llm=llm, prompt=prompt_template)
    st.subheader("Result 1")
    st.info(question_chain.run(user_question))

    template = "Here is a statement:\n{statement}\nMake a bullet point list of the assumptions you made when producing the above statement.\n\n"
    prompt_template = PromptTemplate(
        input_variables=["statement"], template=template)
    assumptions_chain = LLMChain(llm=llm, prompt=prompt_template)
    assumptions_chain_seq = SimpleSequentialChain(
        chains=[question_chain, assumptions_chain], verbose=True
    )
    st.subheader("Result 2")
    st.markdown(assumptions_chain_seq.run(user_question))

    # Chain 3
    template = "Here is a bullet point list of assertions:\n{assertions}\nFor each assertion, determine whether it is true or false. If it is false, explain why.\n\n"
    prompt_template = PromptTemplate(
        input_variables=["assertions"], template=template)
    fact_checker_chain = LLMChain(llm=llm, prompt=prompt_template)
    fact_checker_chain_seq = SimpleSequentialChain(
        chains=[question_chain, assumptions_chain,
                fact_checker_chain], verbose=True
    )
    st.subheader("Result 3")
    st.markdown(fact_checker_chain_seq.run(user_question))

    # Final Chain
    template = "In light of the above facts, how would you answer the question '{}'\n".format(
        user_question)
    template = "{facts}\n" + template
    prompt_template = PromptTemplate(
        input_variables=["facts"], template=template)
    answer_chain = LLMChain(llm=llm, prompt=prompt_template)
    st.subheader("Final Result")
    overall_chain = SimpleSequentialChain(
        chains=[question_chain, assumptions_chain,
                fact_checker_chain, answer_chain],
        verbose=True,
    )

    st.success(overall_chain.run(user_question))
