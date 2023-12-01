import streamlit as st

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.prompts.prompt import PromptTemplate
from langsearch.chains.qa import QAChain


qa_template = """Answer the question as truthfully as possible using the following context, and if the answer is not contained in the context, say "I don't know."
Context:
{context}

Question: {question}
Answer, according to the supplied context: """
qa_prompt = PromptTemplate(template=qa_template, input_variables=["context", "question"])

st.title("Simple Question Answering Demo")
st.text_input("Type your question here", key="simple_question")
qa_chain = load_qa_chain(
    llm=OpenAI(temperature=0, model_name="text-davinci-003"),
    prompt=qa_prompt,
)
question = st.session_state.simple_question
if len(question) != 0:
    chain_output = QAChain(qa_chain=qa_chain)({"question": question})
    answer = chain_output["output_text"]
    st.write(answer)
    for index, doc in enumerate(chain_output["docs"]):
        url = doc.metadata["source"]
        text = doc.page_content
        st.markdown(f"[{index + 1}] [{url}]({url})")
        st.text(text)
