# from GLOBAL_ENV import OPENAI_API_KEY
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import NLTKTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os


def load_split_docs(path: os.PathLike):
    docs = DirectoryLoader(path, glob="*.pdf", loader_cls=PyPDFLoader, use_multithreading=True)
    splits = docs.load_and_split(NLTKTextSplitter(chunk_size= 2048))
    return splits

def create_vectorstore(splits):
    vectorstore = Chroma.from_documents(splits, embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY), persist_directory="vectorstore")
    return vectorstore

def retrieve_and_answer(question):
    template= """
    Provide a detailed, direct and complete answer using the context information. Use the rules bellow:

    1. Provide a complete and in depth answer with the relevant details;
    2. If the informations provided are not enough, answer: "Unfortunately it isn't possible to answer the question due the lack of context. Please rewrite the answer with more details or search the selected document.";
    3. Always provide the source in the format "Title - Section - Page(s)";
    4. For question about multiple topics, break the answer in bullet points;
    5. Separate multiple questions in individual topics.
    6. Answer in the same language of question.

    {context}

    Question: {question}

    Useful answer:

    Source:
    Page(s):
"""
    QA_Prompt = PromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4", temperature=.7, openai_api_key=OPENAI_API_KEY, )
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever= store.as_retriever(),
        chain_type_kwargs={"prompt":QA_Prompt},
    )
    answer = qa_chain({"query":question})
    return answer["result"]

if __name__ == "__main__":
    import dotenv
    splits = load_split_docs("src")
    store = create_vectorstore(splits)
    dotenv.load_dotenv()
    query = str()
    while query.lower() != "exit":
        query = str(input("Question: \n\t"))
        if query.lower != "exit":
            print(f"Answer: \n\t{retrieve_and_answer(query)}")
