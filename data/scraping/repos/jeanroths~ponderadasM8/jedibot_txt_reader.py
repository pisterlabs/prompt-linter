from langchain.llms import Ollama
import gradio as gr
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap, RunnablePassthrough
from langchain.document_loaders import SeleniumURLLoader, DirectoryLoader, TextLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import chroma
from langchain.chains import RetrievalQA

ollama = Ollama(base_url='http://localhost:11434', model='llama2')


def load_archive_and_vectorize():
    loader = TextLoader("text.txt")
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=0)
    docs = splitter.split_documents(documents)

    embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = chroma.Chroma.from_documents(docs, embedding)

    retriever = RetrievalQA.from_chain_type(llm=Ollama(), chain_type="stuff", retriever=vectorstore.as_retriever(search_kwargs={"k":1}))

    return retriever

def run_ollama(input_text):
    prompt = ChatPromptTemplate.from_template(
    """
    You are Yoda from Star Wars. Act as my personal expert on security manager answering in brazilian portuguese any {activity} that user input on chat. {question}
    """
    )
    retriever = load_archive_and_vectorize()
    chain = {"activity": retriever, "question":RunnablePassthrough()} | prompt | ollama
    chat_history = []
    try:
        print(f"Enviando solicitação para OLLAMA: {input_text}")
        msg = ""
        for s in chain.stream(input_text):
            print(s, end="", flush=True)
            msg += s
            yield msg
        chat_history.append((input_text, msg))
        return "", chat_history
    except Exception as e:
        print(f"Error: {e}")
        return "Desculpe, ocorreu um erro ao processar sua solicitação.", chat_history

interface = gr.Interface(
     fn = run_ollama,
     title= "Bem Vindo ao Jedi-SecurityManager Bot",
     inputs = "text",
     outputs = "text",
     description="Converse com o Yoda Security Manager!!"
     ).queue()
''
print("Mestre Yoda quer falar com você...")
interface.launch()
