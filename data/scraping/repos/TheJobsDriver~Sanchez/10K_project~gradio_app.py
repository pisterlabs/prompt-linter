import gradio as gr
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyMuPDFLoader
import os, config

# Set environment variables
os.environ['OPENAI_API_KEY'] = config.OPENAI_API_KEY
os.environ['ORG_ID'] = config.ORG_ID

# Global variables
COUNT, N = 0, 0
chat_history = []
chain = ''
enable_box = gr.Textbox.update(value=None, 
                          placeholder='Upload your OpenAI API key', interactive=True)
disable_box = gr.Textbox.update(value='OpenAI API key is Set', interactive=False)

# Set functions
def set_apikey(api_key):
    os.environ['OPENAI_API_KEY'] = api_key
    return disable_box
    
def enable_api_box():
    return enable_box

# Function to add text to the chat history
def add_text(history, text):
    if not text:
        raise gr.Error('Enter text')
    history = history + [(text, '')]
    return history

# Function to process the PDF file and create a conversation chain
def process_file(file):
    # if 'OPENAI_API_KEY' not in os.environ:
    #     raise gr.Error('Upload your OpenAI API key')
    loader = PyMuPDFLoader(file.name)
    documents = loader.load()

    #embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceEmbeddings()
    pdfsearch = Chroma.from_documents(documents, embeddings)

    chain = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.3, openai_organization=os.getenv("ORG_ID"), model='gpt-4'), 
                                                  retriever=pdfsearch.as_retriever(search_kwargs={"k": 1}),
                                                  return_source_documents=True,)
    
    return chain

# Function to generate a response based on the chat history and query
def generate_response(history, query, btn):
    global COUNT, N, chat_history, chain
    
    if not btn:
        raise gr.Error(message='Upload a PDF')
    if COUNT == 0:
        chain = process_file(btn)
        query = 'Based in the document I passed to you please answer the questions as closely and accurately as possible to the document.'
        COUNT += 1
    
    result = chain({"question": query, 'chat_history': chat_history}, return_only_outputs=True)
    chat_history += [(query, result["answer"])]
    N = list(result['source_documents'][0])[1][1]['page']

    for char in result['answer']:
        history[-1][-1] += char
        yield history, ''

with gr.Blocks() as demo:
    with gr.Column():
        with gr.Row():
            with gr.Column(scale=1):
                api_key = gr.Textbox(
                    placeholder='Enter OpenAI API key',
                    show_label=False,
                    interactive=True, container=False
                )
            with gr.Column(scale=1):
                change_api_key = gr.Button('Change Key')

        with gr.Row():
            chatbot = gr.Chatbot(value=[], elem_id='chatbot', height=650)
            btn = gr.File(label='Upload PDF', height=650)

    with gr.Row():
        with gr.Column(scale=1):
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter", container=False
            )
            submit_btn = gr.Button('Submit')


    enable_box = gr.Textbox.update(value=None,placeholder= 'Upload your OpenAI API key',
                                    interactive=True)
    disable_box = gr.Textbox.update(value = 'OpenAI API key is Set',interactive=False)
    
    # Event handler for submitting the OpenAI API key
    api_key.submit(fn=set_apikey, inputs=[api_key], outputs=[api_key])

    # Event handler for changing the API key
    change_api_key.click(fn=enable_api_box, outputs=[api_key])

    # Event handler for submitting text and generating response
    submit_btn.click(
        fn=add_text,
        inputs=[chatbot, txt],
        outputs=[chatbot],
        queue=False
    ).success(
        fn=generate_response,
        inputs=[chatbot, txt, btn],
        outputs=[chatbot, txt]
    )

demo.queue()
demo.launch()

