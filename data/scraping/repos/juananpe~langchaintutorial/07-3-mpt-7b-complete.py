import typer

# 0xVs
# SOURCE: https://huggingface.co/TheBloke/MPT-7B-Instruct-GGML/discussions/2

from ctransformers.langchain import CTransformers
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from rich import print
from rich.prompt import Prompt

app = typer.Typer()
device = "cpu"

@app.command()
def import_pdfs(dir: str, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
    loader = DirectoryLoader(dir, glob="./*.pdf", loader_cls=PDFPlumberLoader, show_progress=True)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceInstructEmbeddings(model_name=embedding_model, 
                                               model_kwargs={"device": device})
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("faiss_index")

@app.command()
def question(model_path: str = "./models/mpt-7b-instruct.ggmlv3.q5_0.bin",
             model_type='mpt',
             embedding_model="sentence-transformers/all-MiniLM-L6-v2",
             search_breadth : int = 5, threads : int = 6, temperature : float = 0.4):
    embeddings = HuggingFaceInstructEmbeddings(model_name=embedding_model, 
                                               model_kwargs={"device": device})
    config = {'temperature': temperature, 'threads' : threads}
    llm = CTransformers(model=model_path, model_type=model_type, config=config)
    db = FAISS.load_local("faiss_index", embeddings)
    retriever = db.as_retriever(search_kwargs={"k": search_breadth})
    memory = ConversationBufferMemory(memory_key="chat_history", output_key="answer", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever,
                                               memory=memory, return_source_documents=True)
    while True:
        query = Prompt.ask('[bright_yellow]\nQuestion[/bright_yellow] ')
        res = qa({"question": query})
        print("[spring_green4]"+res['answer']+"[/spring_green4]")
        if "source_documents" in res:
            print("\n[italic grey46]References[/italic grey46]:")
            for ref in res["source_documents"]:
                print("> [grey19]" + ref.metadata['source'] + "[/grey19]")

if __name__ == "__main__":
    app()