import streamlit as st

from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.prompts.prompt import PromptTemplate
from langsearch.chains.hyde import HYDEChain
from langsearch.chains.qa import QAChain


hyde_template = """You are an expert in the Python programming language and you like to provide helpful answers to questions. Please answer the following question.
Question: {QUESTION}
Answer:"""
hyde_prompt = PromptTemplate(template=hyde_template, input_variables=["QUESTION"])

qa_template = """Answer the question as truthfully as possible using the following context, and if the answer is not contained in the context, say "I don't know."
Context:
{context}

Question: {question}
Answer, according to the supplied context: """
qa_prompt = PromptTemplate(template=qa_template, input_variables=["context", "question"])

st.title("HYDE Question Answering Demo")
st.text_input("Type your question here", key="hyde_question")
qa_chain = load_qa_chain(
    llm=OpenAI(temperature=0, model_name="text-davinci-003"),
    prompt=qa_prompt,
)
hyde = HYDEChain(
    hyde_llm_chain=LLMChain(
        llm=OpenAI(temperature=0.7, model_name="text-davinci-003"),
        prompt=hyde_prompt
    ),
    langsearch_qa_chain=QAChain(
        top=100,
        qa_chain=qa_chain,
        document_search_question_input_name="hyde_llm_output"
    )
)
question = st.session_state.hyde_question
if len(question) != 0:
    chain_output = hyde({"QUESTION": question})
    answer = chain_output["output_text"]
    st.write(answer)
    for index, doc in enumerate(chain_output["docs"]):
        url = doc.metadata["source"]
        text = doc.page_content
        st.markdown(f"[{index + 1}] [{url}]({url})")
        st.text(text)
