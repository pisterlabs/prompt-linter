# program to create a grapical user interface for the chatbot

# import all necessay langchain modules
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GPT4All
from langchain.vectorstores import FAISS
import gradio as gr
from create_embeddings import create_embeddings_from_pdf

file="Discover_Cardmember_Agreement_Page1.pdf"
chunk_size=1024
chunk_overlap=64
model_name="roberta-base"
embedding_function=HuggingFaceEmbeddings
persist_directory="faiss_dcmap1"



embeddings = embedding_function(
    #model_name="sentence-transformers/all-MiniLM-L6-v2"
    model_name=model_name
)

db_faiss = FAISS.load_local(persist_directory, embeddings)

#load Nous-Hermes model in gpt4all
llm = GPT4All(
    model="./nous-hermes-13b.ggmlv3.q4_0.bin",
    n_ctx=1024,
    backend="nous-hermes",
    verbose=False
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db_faiss.as_retriever(search_type="similarity",search_kwargs={"k":10}),
    #retriever=db.as_retriever(search_kwargs={"k":3}),
    return_source_documents=True,
    verbose=False
)

def generate_answer(input_text):
    #create_embeddings_from_pdf(file=pdf_file)
    return qa(input_text)

# call generate_answer function
#generate_answer(input_text="What is the APR?")

# create a gradio interface
gr.Interface(fn=generate_answer, inputs="textbox", outputs="textbox").launch(enable_queue=True,share=True)