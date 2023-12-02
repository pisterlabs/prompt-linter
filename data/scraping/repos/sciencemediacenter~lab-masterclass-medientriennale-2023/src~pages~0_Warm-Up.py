import streamlit as st

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain

from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI

load_dotenv()

### start page
st.title("🦜⛓️ Warm-Up")
st.caption("Hello, LangChain!")

tab1, tab2 = st.tabs(["Simple Chain", "Sequential Chain"])

with tab1:
    st.caption("Cf. https://python.langchain.com/docs/modules/chains/")
    with st.expander("Code"):
        st.code(
            """
            llm = ChatOpenAI(temperature=0.9, model="gpt-3.5-turbo-0613")

            company_name_prompt = PromptTemplate(
                        template="What is a good name for a company that makes {product}?",
                        input_variables=["product"],
                    )

            company_name_chain = LLMChain(llm=llm, prompt=company_name_prompt)"""
        )

    product = st.text_input("Product name", "colorful socks")
    submit = st.button("Submit", key="simple_chain_submit", type="primary")

    if product and submit:
        company_name_prompt = PromptTemplate(
            template="What is a good name for a company that makes {product}?",
            input_variables=["product"],
        )

        llm = ChatOpenAI(temperature=0.9, model="gpt-3.5-turbo-0613")
        company_name_chain = LLMChain(llm=llm, prompt=company_name_prompt)
        st.info(company_name_chain.run(product))

with tab2:
    st.caption("Cf. https://python.langchain.com/docs/modules/chains/foundational/sequential_chains")
    with st.expander("Code"):
        st.code(
            """
                llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo-0613")

                # This is an LLMChain to write a synopsis given a title of a play.
                synopsis_template = \"\"\"You are a playwright. 
                Given the title of play, it is your job to write a synopsis for that title.

                Title: {title}
                Playwright: This is a synopsis for the above play:\"\"\"
                synopsis_prompt_template = PromptTemplate(input_variables=["title"], 
                                                          template=synopsis_template)
                synopsis_chain = LLMChain(llm=llm, prompt=synopsis_prompt_template)

                # This is an LLMChain to write a review of a play given a synopsis.
                review_template = \"\"\"You are a play critic from the New York Times.
                Given the synopsis of play, it is your job to write a review for that play.

                Play Synopsis:
                {synopsis}
                Review from a New York Times play critic of the above play:\"\"\"
                prompt_template = PromptTemplate(input_variables=["synopsis"], 
                                                 template=review_template)
                review_chain = LLMChain(llm=llm, prompt=prompt_template)

                # This is the overall chain where we run these two chains in sequence.
                overall_chain = SimpleSequentialChain(chains=[synopsis_chain, review_chain], verbose=True)
                """
        )

    play_title = st.text_input("Title of play", "A Parrot in Chains")
    submit = st.button("Submit", key="seq_chain_submit", type="primary")

    if play_title and submit:
        llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo-0613")

        synopsis_template = """You are a playwright. Given the title of play, it is your job to write a synopsis for that title.

        Title: {title}
        Playwright: This is a synopsis for the above play:"""
        synopsis_prompt_template = PromptTemplate(input_variables=["title"], template=synopsis_template)
        synopsis_chain = LLMChain(llm=llm, prompt=synopsis_prompt_template)

        review_template = """You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.

        Play Synopsis:
        {synopsis}
        Review from a New York Times play critic of the above play:"""
        prompt_template = PromptTemplate(input_variables=["synopsis"], template=review_template)
        review_chain = LLMChain(llm=llm, prompt=prompt_template)

        overall_chain = SimpleSequentialChain(chains=[synopsis_chain, review_chain], verbose=True)

        st.info(overall_chain.run(play_title))
