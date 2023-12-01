import os
import PySimpleGUI as sg
from lib.util import create_embedding_vectorstore, load_file, split_text
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

def create_file_conversation_chain(api_key, file_path):
    if api_key is None:
        sg.popup("Please enter your OpenAI API Key!")
        return None
    if file_path is None:
        sg.popup("Please upload a file first!")

    os.environ["OPENAI_API_KEY"] = api_key

    # load file
    loader = load_file(file_path)
    # split text
    texts = split_text(loader)
    # create embedding vectorstore
    vectorstore = create_embedding_vectorstore(texts)
    
    return ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0), vectorstore.as_retriever())

def question_processor(conversation_chain):
    chat_history = []
    def _processor(question):
        answer = conversation_chain({"question": question, 'chat_history': chat_history})
        chat_history.append((question, answer['answer']))
        
        return "\n".join([f"Q: {q}\nA: {a}" for q, a in chat_history])
    return _processor

# Create the layout of the window
layout = [
    [sg.Text("Enter your OpenAI API Key:")],
    [sg.InputText(key="-KEY_TEXT-")],
    [sg.Text("Select a PDF file:")],
    [sg.Input(key="-FILE-"), sg.FileBrowse(file_types=(("PDF Files", "*.pdf", "*.txt"),)), sg.Button("Upload")],
    [sg.Multiline(key="-ROLLER-", size=(100, 40), enable_events=True, autoscroll=True, reroute_stdout=True)],
    [sg.InputText(key="-INPUT-", size=(90, 2)), sg.Button("Send")]
]

# Create the window
window = sg.Window("ChatPDF", layout, size=(700, 600))

# Initialize variables
processor = None

# Event loop to process window events
while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED or event == "Cancel":
        break
    elif event == "Upload":
        api_key = values["-KEY_TEXT-"]
        file_name = values["-FILE-"]
        qa_chain = create_file_conversation_chain(api_key, file_name)
        processor = question_processor(qa_chain)
        
    elif event == "Send":
        # Raise an alert if no PDF file uploaded
        if not processor:
            sg.popup("Please init a qa processor by upload a file and enter your OpenAI API Key!")
            continue
        
        question = values["-INPUT-"]
        result = processor(question)
        
        window["-ROLLER-"].update(result)
        window["-INPUT-"].update("")

# Close the window
window.close()