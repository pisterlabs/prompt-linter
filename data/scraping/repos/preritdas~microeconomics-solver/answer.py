"""Answer questions based on the textbook."""
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

from textbook.embed import load_vectorstore
from keys import KEYS


prompt_template = """Use the following pieces of context from a Microeconomics textbook to answer the question at the end. 
The textbook is the 9th edition of _Microeconomics_ by Pindyck and Rubinfeld.
Only answer the question with information from the textbook.
If you don't know the answer, meaning it wasn't in the contexts, just say that you don't know, DO NOT make up an answer.
Each piece of context has a page number at the end. Always cite your sources by providing this page number when the context is used. Include snippets from the textbook contexts where appropriate.

=======
Start of Context
=======

{context}

=======
End of Context
=======

Question: {question}
Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

llm = ChatOpenAI(model="gpt-4", openai_api_key=KEYS.OpenAI.api_key)
chain_type_kwargs = {"prompt": PROMPT}
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=load_vectorstore().as_retriever(), chain_type_kwargs=chain_type_kwargs)


def ask_textbook(question: str) -> str:
    """Ask the textbook a question."""
    return qa.run(question)
