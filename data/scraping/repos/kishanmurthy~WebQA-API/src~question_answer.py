from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer: """


def get_qa_result(webpage: str, question: str) -> str:
    '''
    Returns the answer to the question from the webpage
    '''
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(webpage)
    embeddings = OpenAIEmbeddings()

    docsearch = Chroma.from_texts(texts, embeddings, metadatas=[{"source": str(i)} for i in range(len(texts))]).as_retriever()

    docs = docsearch.get_relevant_documents(question)


    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"])


    chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff", prompt=PROMPT)
    return chain({"input_documents": docs, "question": question}, return_only_outputs=True)



