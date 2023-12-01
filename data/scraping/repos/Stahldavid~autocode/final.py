# final
import gradio as gr
import zipfile
import shutil
import pandas as pd
from langchain.chains import ConversationalRetrievalChain

def create_dataframe(source_documents):
    data = []
    for doc in source_documents:
        page_num = doc.metadata['page']
        doc_name = doc.metadata['source']
        data.append({'Page Number': page_num, 'Document Name': doc_name})
    df = pd.DataFrame(data)
    return df.to_string()

def unir_textos_documentos(source_documents):
    textos = [documento.page_content for documento in source_documents]
    texto_unido = ' '.join(textos)
    return texto_unido

index = None
source_documents_chatbot_messages = []

def chat(chat_history, message):
    global source_documents_chatbot_messages
    qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0), index.as_retriever(n_docs=2), return_source_documents=True)
    # Get answer from the model
    result = qa({"question": message, "chat_history": chat_history})
    answer = result["answer"]
    chat_history.append((message, answer))
    source_documents = result['source_documents']
    source_documents_text = unir_textos_documentos(source_documents)
    df_string = create_dataframe(source_documents)

    return chat_history, source_documents_text, df_string




def build_the_bot(upload_arquivos):
    dbb_folder = 'dbb'
    shutil.rmtree(dbb_folder, ignore_errors=True)
    with zipfile.ZipFile(upload_arquivos.name, 'r') as zip_ref:
        zip_ref.extractall(dbb_folder)
    global index
    index = Chroma(persist_directory=dbb_folder, embedding_function=embeddings)
    
    chat_history = [("Bot", "Index saved successfully!!!")]
    return chat_history

def clear_chat_history(chatbot):
    chatbot.clear()
    chat_history = []
    return chat_history

with gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}") as demo:
    state = gr.State(source_documents="")

    with gr.Row():
        with gr.Column(scale=0.75):
            with gr.Tab("Chatbot"):
                chatbot = gr.Chatbot([], elem_id="chatbot").style(height=750)
                messagem = gr.Textbox(show_label=False, placeholder="Enter text and press enter")
        with gr.Column(scale=0.5):
            with gr.Tab("Source Documents"):
                with gr.Row():
                    source_documents_chatbot = gr.Textbox([], elem_id="source_documents_text").style(height=750)
                with gr.Row():
                    df_textbox = gr.Textbox([], elem_id="df_textbox").style(height=250)

    messagem.submit(chat, [chatbot, messagem], [chatbot, source_documents_chatbot, df_textbox])

    messagem.submit(lambda: "", None, messagem)

    with gr.Row():
        with gr.Column(scale=0.85):
            btn = gr.UploadButton("📁", directory=True)
            btn.upload(build_the_bot, btn, chatbot)
        with gr.Column(scale=0.15, min_width=0):
            clear_btn = gr.Button("Clear")
            clear_btn.click(lambda: None, None, chatbot, queue=False)

demo.launch(debug=True)



