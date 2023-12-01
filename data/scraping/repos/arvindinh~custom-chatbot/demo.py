import sys
sys.path.append('../ai/embeddings')
sys.path.append('..')
from openai_embeddings import OpenAI_Embeddings
from loaders.pymupdf import PyMuPDF_Loader
from splitters.recursive import RecursiveCharacter_TextSplitter
from vectorstores.deep_lake import DeeplakeDB
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

def main():
    pdf = '../../ePortG11InstallGuide.pdf'
    # load PDF document, extract text
    loader = PyMuPDF_Loader(pdf)
    data = loader.load_text()
    # split extracted text(tokenize)
    # split recursively by different characters - starting with "\n\n", then "\n", then " "
    splitter = RecursiveCharacter_TextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    docs = [splitter.split_data(data)]
    # initialize embeddings model to pass in to db
    embeddings = OpenAI_Embeddings(api_key='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').vectorizer
    # initialize vector store, add split docs
    # (db will compute embeddings using embedding model and store in specified path)
    deeplake = DeeplakeDB(store_path='training/embeddings_deeplake', embedding_model=embeddings)
    deeplake.add_docs(docs)
    llm = OpenAI(temperature=0,openai_api_key='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')
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=deeplake.db.as_retriever(), memory=memory)
    query = ""
    while True:
        query = input("Ask me a question: ")
        if query.lower() == "exit":
            break
        print("\n")
        result = qa({"question": query})
        print(result["answer"])
        print("\n")

    deeplake.delete_all()

if __name__ == '__main__':
    main()