from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain, ChatVectorDBChain
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores.faiss import FAISS

_template = """Given the following chat history and a follow up question, rephrase the follow up question to be a standalone question.
You can assume that the question is about Flyte.

Chat History:
{chat_history}
Follow Up Question:
{question}
Standalone question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """You are a maintainer developing the open source library Flyte and understanding the codebase very well.
You are given the following extracted parts of the context and a question. Provide a conversational answer in a concise and clear manner. Attach a link if neccessary.
Please answer based on the question.

Question: {question}
=========
Context:
{context}
=========
Answer in Markdown:"""

QA_PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])


def get_chain(vectorstore):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        vectorstore.as_retriever(search_kwargs={"k": 20}),
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={'prompt': QA_PROMPT},
        # verbose=True,
    )
    return qa_chain

def start_conversation(vectorstore):
    qa_chain = get_chain(vectorstore)
    chat_history = []
    print("Chat with your docs!")
    while True:
        print("Human:")
        question = input()
        print(question)
        result = qa_chain({"question": question, "chat_history": chat_history})
        chat_history.append((question, result["answer"]))
        print("AI:")
        print(result["answer"])
