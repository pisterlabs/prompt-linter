# Imports
import base64
import glob
import json
import math
import openai
import os
import pytz
import re
import requests
import streamlit as st
import textract
import time
import zipfile
import huggingface_hub
import dotenv
from audio_recorder_streamlit import audio_recorder
from bs4 import BeautifulSoup
from collections import deque
from datetime import datetime
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from io import BytesIO
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from openai import ChatCompletion
from PyPDF2 import PdfReader
from templates import bot_template, css, user_template
from xml.etree import ElementTree as ET
import streamlit.components.v1 as components  # Import Streamlit Components for HTML5


st.set_page_config(page_title="🐪Llama Whisperer🦙 Voice Chat🌟", layout="wide")


def add_Med_Licensing_Exam_Dataset():
    import streamlit as st
    from datasets import load_dataset
    dataset = load_dataset("augtoma/usmle_step_1")['test']  # Using 'test' split
    st.title("USMLE Step 1 Dataset Viewer")
    if len(dataset) == 0:
        st.write("😢 The dataset is empty.")
    else:
        st.write("""
        🔍 Use the search box to filter questions or use the grid to scroll through the dataset.
        """)
    
        # 👩‍🔬 Search Box
        search_term = st.text_input("Search for a specific question:", "")
        
        # 🎛 Pagination
        records_per_page = 100
        num_records = len(dataset)
        num_pages = max(int(num_records / records_per_page), 1)
        
        # Skip generating the slider if num_pages is 1 (i.e., all records fit in one page)
        if num_pages > 1:
            page_number = st.select_slider("Select page:", options=list(range(1, num_pages + 1)))
        else:
            page_number = 1  # Only one page
        
        # 📊 Display Data
        start_idx = (page_number - 1) * records_per_page
        end_idx = start_idx + records_per_page
    
        # 🧪 Apply the Search Filter
        filtered_data = []
        for record in dataset[start_idx:end_idx]:
            if isinstance(record, dict) and 'text' in record and 'id' in record:
                if search_term:
                    if search_term.lower() in record['text'].lower():
                        filtered_data.append(record)
                else:
                    filtered_data.append(record)
    
        # 🌐 Render the Grid
        for record in filtered_data:
            st.write(f"## Question ID: {record['id']}")
            st.write(f"### Question:")
            st.write(f"{record['text']}")
            st.write(f"### Answer:")
            st.write(f"{record['answer']}")
            st.write("---")
    
        st.write(f"😊 Total Records: {num_records} | 📄 Displaying {start_idx+1} to {min(end_idx, num_records)}")

# 1. Constants and Top Level UI Variables

# My Inference API Copy
# API_URL = 'https://qe55p8afio98s0u3.us-east-1.aws.endpoints.huggingface.cloud'  # Dr Llama
# Original:
API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
API_KEY = os.getenv('API_KEY')
MODEL1="meta-llama/Llama-2-7b-chat-hf"
MODEL1URL="https://huggingface.co/meta-llama/Llama-2-7b-chat-hf"
HF_KEY = os.getenv('HF_KEY')
headers = {
    "Authorization": f"Bearer {HF_KEY}",
    "Content-Type": "application/json"
}
key = os.getenv('OPENAI_API_KEY')
prompt = f"Write instructions to teach anyone to write a discharge plan. List the entities, features and relationships to CCDA and FHIR objects in boldface."
should_save = st.sidebar.checkbox("💾 Save", value=True, help="Save your session data.")

# 2. Prompt label button demo for LLM
def add_witty_humor_buttons():
    with st.expander("Wit and Humor 🤣", expanded=True):
        # Tip about the Dromedary family
        st.markdown("🔬 **Fun Fact**: Dromedaries, part of the camel family, have a single hump and are adapted to arid environments. Their 'superpowers' include the ability to survive without water for up to 7 days, thanks to their specialized blood cells and water storage in their hump.")
        
        # Define button descriptions
        descriptions = {
            "Generate Limericks 😂": "Write ten random adult limericks based on quotes that are tweet length and make you laugh 🎭",
            "Wise Quotes 🧙": "Generate ten wise quotes that are tweet length 🦉",
            "Funny Rhymes 🎤": "Create ten funny rhymes that are tweet length 🎶",
            "Medical Jokes 💉": "Create ten medical jokes that are tweet length 🏥",
            "Minnesota Humor ❄️": "Create ten jokes about Minnesota that are tweet length 🌨️",
            "Top Funny Stories 📖": "Create ten funny stories that are tweet length 📚",
            "More Funny Rhymes 🎙️": "Create ten more funny rhymes that are tweet length 🎵"
        }
        
        # Create columns
        col1, col2, col3 = st.columns([1, 1, 1], gap="small")
        
        # Add buttons to columns
        if col1.button("Generate Limericks 😂"):
            StreamLLMChatResponse(descriptions["Generate Limericks 😂"])
        
        if col2.button("Wise Quotes 🧙"):
            StreamLLMChatResponse(descriptions["Wise Quotes 🧙"])
        
        if col3.button("Funny Rhymes 🎤"):
            StreamLLMChatResponse(descriptions["Funny Rhymes 🎤"])
        
        col4, col5, col6 = st.columns([1, 1, 1], gap="small")
        
        if col4.button("Medical Jokes 💉"):
            StreamLLMChatResponse(descriptions["Medical Jokes 💉"])
        
        if col5.button("Minnesota Humor ❄️"):
            StreamLLMChatResponse(descriptions["Minnesota Humor ❄️"])
        
        if col6.button("Top Funny Stories 📖"):
            StreamLLMChatResponse(descriptions["Top Funny Stories 📖"])
        
        col7 = st.columns(1, gap="small")
        
        if col7[0].button("More Funny Rhymes 🎙️"):
            StreamLLMChatResponse(descriptions["More Funny Rhymes 🎙️"])

def SpeechSynthesis(result):
    documentHTML5='''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Read It Aloud</title>
        <script type="text/javascript">
            function readAloud() {
                const text = document.getElementById("textArea").value;
                const speech = new SpeechSynthesisUtterance(text);
                window.speechSynthesis.speak(speech);
            }
        </script>
    </head>
    <body>
        <h1>🔊 Read It Aloud</h1>
        <textarea id="textArea" rows="10" cols="80">
    '''
    documentHTML5 = documentHTML5 + result
    documentHTML5 = documentHTML5 + '''
        </textarea>
        <br>
        <button onclick="readAloud()">🔊 Read Aloud</button>
    </body>
    </html>
    '''

    components.html(documentHTML5, width=1280, height=1024)
    #return result


# 3. Stream Llama Response
# @st.cache_resource
def StreamLLMChatResponse(prompt):
    try:
        endpoint_url = API_URL
        hf_token = API_KEY
        client = InferenceClient(endpoint_url, token=hf_token)
        gen_kwargs = dict(
            max_new_tokens=512,
            top_k=30,
            top_p=0.9,
            temperature=0.2,
            repetition_penalty=1.02,
            stop_sequences=["\nUser:", "<|endoftext|>", "</s>"],
        )
        stream = client.text_generation(prompt, stream=True, details=True, **gen_kwargs)
        report=[]
        res_box = st.empty()
        collected_chunks=[]
        collected_messages=[]
        allresults=''
        for r in stream:
            if r.token.special:
                continue
            if r.token.text in gen_kwargs["stop_sequences"]:
                break
            collected_chunks.append(r.token.text)
            chunk_message = r.token.text
            collected_messages.append(chunk_message)
            try:
                report.append(r.token.text)
                if len(r.token.text) > 0:
                    result="".join(report).strip()
                    res_box.markdown(f'*{result}*')
                    
            except:
                st.write('Stream llm issue')
        SpeechSynthesis(result)
        return result
    except:
        st.write('Llama model is asleep. Starting up now on A10 - please give 5 minutes then retry as KEDA scales up from zero to activate running container(s).')

# 4. Run query with payload
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    st.markdown(response.json())
    return response.json()
def get_output(prompt):
    return query({"inputs": prompt})

# 5. Auto name generated output files from time and content
def generate_filename(prompt, file_type):
    central = pytz.timezone('US/Central')
    safe_date_time = datetime.now(central).strftime("%m%d_%H%M")
    replaced_prompt = prompt.replace(" ", "_").replace("\n", "_")
    safe_prompt = "".join(x for x in replaced_prompt if x.isalnum() or x == "_")[:45]
    return f"{safe_date_time}_{safe_prompt}.{file_type}"

# 6. Speech transcription via OpenAI service
def transcribe_audio(openai_key, file_path, model):
    openai.api_key = openai_key
    OPENAI_API_URL = "https://api.openai.com/v1/audio/transcriptions"
    headers = {
        "Authorization": f"Bearer {openai_key}",
    }
    with open(file_path, 'rb') as f:
        data = {'file': f}
        response = requests.post(OPENAI_API_URL, headers=headers, files=data, data={'model': model})
    if response.status_code == 200:
        st.write(response.json())
        chatResponse = chat_with_model(response.json().get('text'), '') # *************************************
        transcript = response.json().get('text')
        filename = generate_filename(transcript, 'txt')
        response = chatResponse
        user_prompt = transcript
        create_file(filename, user_prompt, response, should_save)
        return transcript
    else:
        st.write(response.json())
        st.error("Error in API call.")
        return None

# 7. Auto stop on silence audio control for recording WAV files
def save_and_play_audio(audio_recorder):
    audio_bytes = audio_recorder(key='audio_recorder')
    if audio_bytes:
        filename = generate_filename("Recording", "wav")
        with open(filename, 'wb') as f:
            f.write(audio_bytes)
        st.audio(audio_bytes, format="audio/wav")
        return filename
    return None

# 8. File creator that interprets type and creates output file for text, markdown and code
def create_file(filename, prompt, response, should_save=True):
    if not should_save:
        return
    base_filename, ext = os.path.splitext(filename)
    if ext in ['.txt', '.htm', '.md']:
        with open(f"{base_filename}.md", 'w') as file:
            try:
                content = prompt.strip() + '\r\n' + response
                file.write(content)
            except:
                st.write('.')

    #has_python_code = re.search(r"```python([\s\S]*?)```", prompt.strip() + '\r\n' + response)
    #has_python_code = bool(re.search(r"```python([\s\S]*?)```", prompt.strip() + '\r\n' + response))
        #if has_python_code:
        #    python_code = re.findall(r"```python([\s\S]*?)```", response)[0].strip()
        #    with open(f"{base_filename}-Code.py", 'w') as file:
        #        file.write(python_code)
        #    with open(f"{base_filename}.md", 'w') as file:
        #        content = prompt.strip() + '\r\n' + response
        #        file.write(content)
            
def truncate_document(document, length):
    return document[:length]
def divide_document(document, max_length):
    return [document[i:i+max_length] for i in range(0, len(document), max_length)]

# 9. Sidebar with UI controls to review and re-run prompts and continue responses
@st.cache_resource
def get_table_download_link(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
   
    b64 = base64.b64encode(data.encode()).decode()  
    file_name = os.path.basename(file_path)
    ext = os.path.splitext(file_name)[1]  # get the file extension
    if ext == '.txt':
        mime_type = 'text/plain'
    elif ext == '.py':
        mime_type = 'text/plain'
    elif ext == '.xlsx':
        mime_type = 'text/plain'
    elif ext == '.csv':
        mime_type = 'text/plain'
    elif ext == '.htm':
        mime_type = 'text/html'
    elif ext == '.md':
        mime_type = 'text/markdown'
    else:
        mime_type = 'application/octet-stream'  # general binary data type
    href = f'<a href="data:{mime_type};base64,{b64}" target="_blank" download="{file_name}">{file_name}</a>'
    return href


def CompressXML(xml_text):
    root = ET.fromstring(xml_text)
    for elem in list(root.iter()):
        if isinstance(elem.tag, str) and 'Comment' in elem.tag:
            elem.parent.remove(elem)
    return ET.tostring(root, encoding='unicode', method="xml")

# 10. Read in and provide UI for past files
@st.cache_resource
def read_file_content(file,max_length):
    if file.type == "application/json":
        content = json.load(file)
        return str(content)
    elif file.type == "text/html" or file.type == "text/htm":
        content = BeautifulSoup(file, "html.parser")
        return content.text
    elif file.type == "application/xml" or file.type == "text/xml":
        tree = ET.parse(file)
        root = tree.getroot()
        xml = CompressXML(ET.tostring(root, encoding='unicode'))
        return xml
    elif file.type == "text/markdown" or file.type == "text/md":
        md = mistune.create_markdown()
        content = md(file.read().decode())
        return content
    elif file.type == "text/plain":
        return file.getvalue().decode()
    else:
        return ""

# 11. Chat with GPT - Caution on quota - now favoring fastest AI pipeline STT Whisper->LLM Llama->TTS
@st.cache_resource
def chat_with_model(prompt, document_section, model_choice='gpt-3.5-turbo'):
    model = model_choice
    conversation = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
    conversation.append({'role': 'user', 'content': prompt})
    if len(document_section)>0:
        conversation.append({'role': 'assistant', 'content': document_section})
    start_time = time.time()
    report = []
    res_box = st.empty()
    collected_chunks = []
    collected_messages = []
    for chunk in openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=conversation, temperature=0.5, stream=True):
        collected_chunks.append(chunk)  
        chunk_message = chunk['choices'][0]['delta']  
        collected_messages.append(chunk_message) 
        content=chunk["choices"][0].get("delta",{}).get("content")
        try:
            report.append(content)
            if len(content) > 0:
                result = "".join(report).strip()
                res_box.markdown(f'*{result}*') 
        except:
            st.write(' ')
    full_reply_content = ''.join([m.get('content', '') for m in collected_messages])
    st.write("Elapsed time:")
    st.write(time.time() - start_time)
    return full_reply_content

# 12. Embedding VectorDB for LLM query of documents to text to compress inputs and prompt together as Chat memory using Langchain
@st.cache_resource
def chat_with_file_contents(prompt, file_content, model_choice='gpt-3.5-turbo'):
    conversation = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
    conversation.append({'role': 'user', 'content': prompt})
    if len(file_content)>0:
        conversation.append({'role': 'assistant', 'content': file_content})
    response = openai.ChatCompletion.create(model=model_choice, messages=conversation)
    return response['choices'][0]['message']['content']

def extract_mime_type(file):
    if isinstance(file, str):
        pattern = r"type='(.*?)'"
        match = re.search(pattern, file)
        if match:
            return match.group(1)
        else:
            raise ValueError(f"Unable to extract MIME type from {file}")
    elif isinstance(file, streamlit.UploadedFile):
        return file.type
    else:
        raise TypeError("Input should be a string or a streamlit.UploadedFile object")

def extract_file_extension(file):
    # get the file name directly from the UploadedFile object
    file_name = file.name
    pattern = r".*?\.(.*?)$"
    match = re.search(pattern, file_name)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Unable to extract file extension from {file_name}")

# Normalize input as text from PDF and other formats
@st.cache_resource
def pdf2txt(docs):
    text = ""
    for file in docs:
        file_extension = extract_file_extension(file)
        st.write(f"File type extension: {file_extension}")
        if file_extension.lower() in ['py', 'txt', 'html', 'htm', 'xml', 'json']:
            text += file.getvalue().decode('utf-8')
        elif file_extension.lower() == 'pdf':
            from PyPDF2 import PdfReader
            pdf = PdfReader(BytesIO(file.getvalue()))
            for page in range(len(pdf.pages)):
                text += pdf.pages[page].extract_text() # new PyPDF2 syntax
    return text

def txt2chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(text)

# Vector Store using FAISS
@st.cache_resource
def vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=key)
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

# Memory and Retrieval chains
@st.cache_resource
def get_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)

def process_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        template = user_template if i % 2 == 0 else bot_template
        st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        filename = generate_filename(user_question, 'txt')
        response = message.content
        user_prompt = user_question
        create_file(filename, user_prompt, response, should_save)       

def divide_prompt(prompt, max_length):
    words = prompt.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        if len(word) + current_length <= max_length:
            current_length += len(word) + 1 
            current_chunk.append(word)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
    chunks.append(' '.join(current_chunk))
    return chunks

    
# 13. Provide way of saving all and deleting all to give way of reviewing output and saving locally before clearing it
    
@st.cache_resource
def create_zip_of_files(files):
    zip_name = "all_files.zip"
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        for file in files:
            zipf.write(file)
    return zip_name
    
@st.cache_resource
def get_zip_download_link(zip_file):
    with open(zip_file, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/zip;base64,{b64}" download="{zip_file}">Download All</a>'
    return href

# 14. Inference Endpoints for Whisper (best fastest STT) on NVIDIA T4 and Llama (best fastest AGI LLM) on NVIDIA A10
# My Inference Endpoint
API_URL_IE = f'https://tonpixzfvq3791u9.us-east-1.aws.endpoints.huggingface.cloud'
# Original
API_URL_IE = "https://api-inference.huggingface.co/models/openai/whisper-small.en"
MODEL2 = "openai/whisper-small.en"
MODEL2_URL = "https://huggingface.co/openai/whisper-small.en"
#headers = {
#	"Authorization": "Bearer XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
#	"Content-Type": "audio/wav"
#}
HF_KEY = os.getenv('HF_KEY')
headers = {
    "Authorization": f"Bearer {HF_KEY}",
    "Content-Type": "audio/wav"
}

#@st.cache_resource
def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL_IE, headers=headers, data=data)
    return response.json()

def generate_filename(prompt, file_type):
    central = pytz.timezone('US/Central')
    safe_date_time = datetime.now(central).strftime("%m%d_%H%M")
    replaced_prompt = prompt.replace(" ", "_").replace("\n", "_")
    safe_prompt = "".join(x for x in replaced_prompt if x.isalnum() or x == "_")[:90]
    return f"{safe_date_time}_{safe_prompt}.{file_type}"

# 15. Audio recorder to Wav file 
def save_and_play_audio(audio_recorder):
    audio_bytes = audio_recorder()
    if audio_bytes:
        filename = generate_filename("Recording", "wav")
        with open(filename, 'wb') as f:
            f.write(audio_bytes)
        st.audio(audio_bytes, format="audio/wav")
        return filename

# 16. Speech transcription to file output
def transcribe_audio(filename):
    output = query(filename)
    return output

def whisper_main():
    st.title("Speech to Text")
    st.write("Record your speech and get the text.")

    # Audio, transcribe, GPT:
    filename = save_and_play_audio(audio_recorder)
    if filename is not None:
        transcription = transcribe_audio(filename)
        try:
            transcription = transcription['text']
        except:
            st.write('Whisper model is asleep. Starting up now on T4 GPU - please give 5 minutes then retry as it scales up from zero to activate running container(s).')

        st.write(transcription)
        response = StreamLLMChatResponse(transcription)
        # st.write(response) - redundant with streaming result?
        filename = generate_filename(transcription, ".txt")
        create_file(filename, transcription, response, should_save)
        #st.sidebar.markdown(get_table_download_link(filename), unsafe_allow_html=True)


# 17. Main
def main():

    st.title("AI Drome Llama")
    prompt = f"Write ten funny jokes that are tweet length stories that make you laugh.  Show as markdown outline with emojis for each."

    # Add Wit and Humor buttons
    add_witty_humor_buttons()

    example_input = st.text_input("Enter your example text:", value=prompt, help="Enter text to get a response from DromeLlama.")
    if st.button("Run Prompt With DromeLlama", help="Click to run the prompt."):
        try:
            StreamLLMChatResponse(example_input)
        except:
            st.write('DromeLlama is asleep. Starting up now on A10 - please give 5 minutes then retry as KEDA scales up from zero to activate running container(s).')

    openai.api_key = os.getenv('OPENAI_KEY')
    menu = ["txt", "htm", "xlsx", "csv", "md", "py"]
    choice = st.sidebar.selectbox("Output File Type:", menu)
    model_choice = st.sidebar.radio("Select Model:", ('gpt-3.5-turbo', 'gpt-3.5-turbo-0301'))        
    user_prompt = st.text_area("Enter prompts, instructions & questions:", '', height=100)
    collength, colupload = st.columns([2,3])  # adjust the ratio as needed
    with collength:
        max_length = st.slider("File section length for large files", min_value=1000, max_value=128000, value=12000, step=1000)
    with colupload:
        uploaded_file = st.file_uploader("Add a file for context:", type=["pdf", "xml", "json", "xlsx", "csv", "html", "htm", "md", "txt"])
    document_sections = deque()
    document_responses = {}
    if uploaded_file is not None:
        file_content = read_file_content(uploaded_file, max_length)
        document_sections.extend(divide_document(file_content, max_length))
    if len(document_sections) > 0:
        if st.button("👁️ View Upload"):
            st.markdown("**Sections of the uploaded file:**")
            for i, section in enumerate(list(document_sections)):
                st.markdown(f"**Section {i+1}**\n{section}")
        st.markdown("**Chat with the model:**")
        for i, section in enumerate(list(document_sections)):
            if i in document_responses:
                st.markdown(f"**Section {i+1}**\n{document_responses[i]}")
            else:
                if st.button(f"Chat about Section {i+1}"):
                    st.write('Reasoning with your inputs...')
                    response = chat_with_model(user_prompt, section, model_choice)
                    st.write('Response:')
                    st.write(response)
                    document_responses[i] = response
                    filename = generate_filename(f"{user_prompt}_section_{i+1}", choice)
                    create_file(filename, user_prompt, response, should_save)
                    st.sidebar.markdown(get_table_download_link(filename), unsafe_allow_html=True)
    if st.button('💬 Chat'):
        st.write('Reasoning with your inputs...')
        user_prompt_sections = divide_prompt(user_prompt, max_length)
        full_response = ''
        for prompt_section in user_prompt_sections:
            response = chat_with_model(prompt_section, ''.join(list(document_sections)), model_choice)
            full_response += response + '\n'  # Combine the responses
        response = full_response
        st.write('Response:')
        st.write(response)
        filename = generate_filename(user_prompt, choice)
        create_file(filename, user_prompt, response, should_save)
        st.sidebar.markdown(get_table_download_link(filename), unsafe_allow_html=True)

    # Compose a file sidebar of past encounters
    all_files = glob.glob("*.*")
    all_files = [file for file in all_files if len(os.path.splitext(file)[0]) >= 20]  # exclude files with short names
    all_files.sort(key=lambda x: (os.path.splitext(x)[1], x), reverse=True)  # sort by file type and file name in descending order
    if st.sidebar.button("🗑 Delete All"):
        for file in all_files:
            os.remove(file)
        st.experimental_rerun()
    if st.sidebar.button("⬇️ Download All"):
        zip_file = create_zip_of_files(all_files)
        st.sidebar.markdown(get_zip_download_link(zip_file), unsafe_allow_html=True)
    file_contents=''
    next_action=''
    for file in all_files:
        col1, col2, col3, col4, col5 = st.sidebar.columns([1,6,1,1,1])  # adjust the ratio as needed
        with col1:
            if st.button("🌐", key="md_"+file):  # md emoji button
                with open(file, 'r') as f:
                    file_contents = f.read()
                    next_action='md'
        with col2:
            st.markdown(get_table_download_link(file), unsafe_allow_html=True)
        with col3:
            if st.button("📂", key="open_"+file):  # open emoji button
                with open(file, 'r') as f:
                    file_contents = f.read()
                    next_action='open'
        with col4:
            if st.button("🔍", key="read_"+file):  # search emoji button
                with open(file, 'r') as f:
                    file_contents = f.read()
                    next_action='search'
        with col5:
            if st.button("🗑", key="delete_"+file):
                os.remove(file)
                st.experimental_rerun()

                
    if len(file_contents) > 0:
        if next_action=='open':
            file_content_area = st.text_area("File Contents:", file_contents, height=500)
        if next_action=='md':
            st.markdown(file_contents)
        if next_action=='search':
            file_content_area = st.text_area("File Contents:", file_contents, height=500)
            st.write('Reasoning with your inputs...')

            # new - llama
            response = StreamLLMChatResponse(file_contents)
            filename = generate_filename(user_prompt, ".md")
            create_file(filename, file_contents, response, should_save)
            SpeechSynthesis(response)
            
            # old - gpt
            #response = chat_with_model(user_prompt, file_contents, model_choice)
            #filename = generate_filename(file_contents, choice)
            #create_file(filename, user_prompt, response, should_save)
            
            st.experimental_rerun()

    # Feedback
    # Step: Give User a Way to Upvote or Downvote
    feedback = st.radio("Step 8: Give your feedback", ("👍 Upvote", "👎 Downvote"))
    if feedback == "👍 Upvote":
        st.write("You upvoted 👍. Thank you for your feedback!")
    else:
        st.write("You downvoted 👎. Thank you for your feedback!")
        
    load_dotenv()
    st.write(css, unsafe_allow_html=True)
    st.header("Chat with documents :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        process_user_input(user_question)
    with st.sidebar:
        st.subheader("Your documents")
        docs = st.file_uploader("import documents", accept_multiple_files=True)
        with st.spinner("Processing"):
            raw = pdf2txt(docs)
            if len(raw) > 0:
                length = str(len(raw))
                text_chunks = txt2chunks(raw)
                vectorstore = vector_store(text_chunks)
                st.session_state.conversation = get_chain(vectorstore)
                st.markdown('# AI Search Index of Length:' + length + ' Created.')  # add timing
                filename = generate_filename(raw, 'txt')
                create_file(filename, raw, '', should_save)

# 18. Run AI Pipeline
if __name__ == "__main__":
    whisper_main()
    main()