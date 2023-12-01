import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import TextLoader


# build environment
def load_log(log):
    os.environ["OPENAI_API_KEY"] = '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'

    #load
    print('reading text')
    loader = TextLoader(log)
    print('loading log')
    data = loader.load()

    
    #split
    print('spliting')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 60, chunk_overlap = 0)
    all_splits = text_splitter.split_documents(data)
    
    #store
    global vectorstore
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
    
    return 1
   
    # docs = vectorstore.similarity_search(question)
def qa_langchain(question):
    
    
    # generate answer
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectorstore.as_retriever())
    
    
    return qa_chain.run({"query": question})


if __name__ == "__main__":
    
    load_log('chat_with_ai/test_log1.txt')
    print('finish loading pdf')
    while True:
        print('\n')
        question= input('Enter your question:')
        print(qa_langchain(question))