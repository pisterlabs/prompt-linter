import os
import pandas as pd
import matplotlib.pyplot as plt
#from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader,DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
import textract
import vertexai
from langchain.llms import VertexAI
from langchain.embeddings import VertexAIEmbeddings
import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import DocArrayInMemorySearch
import os
from google.api_core.client_options import ClientOptions
from google.cloud import documentai
from google.cloud import translate_v2 as translate
import google.cloud.texttospeech as tts


st.sidebar.title("Config")
supported_languages = {
    "af": "Afrikaans",
    "sq": "Albanian",
    "am": "Amharic",
    "ar": "Arabic",
    "az": "Azerbaijani",
    "eu": "Basque",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "ca": "Catalan",
    "zh-CN": "Chinese (Simplified)",
    "zh-TW": "Chinese (Traditional)",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "et": "Estonian",
    "tl": "Filipino",
    "fi": "Finnish",
    "fr": "French",
    "gl": "Galician",
    "ka": "Georgian",
    "de": "German",
    "el": "Greek",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "hu": "Hungarian",
    "is": "Icelandic",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "ko": "Korean",
    "ku": "Kurdish (Kurmanji)",
    "ky": "Kyrgyz",
    "lo": "Lao",
    "la": "Latin",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "mk": "Macedonian",
    "ms": "Malay",
    "ml": "Malayalam",
    "mn": "Mongolian",
    "ne": "Nepali",
    "no": "Norwegian",
    "pa": "Punjabi",
    "fa": "Persian",
    "pl": "Polish",
    "pt-BR": "Portuguese (Brazil)",
    "pt-PT": "Portuguese (Portugal)",
    "ro": "Romanian",
    "ru": "Russian",
    "sr": "Serbian",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovenian",
    "es": "Spanish",
    "su": "Sundanese",
    "sw": "Swahili",
    "sv": "Swedish",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "cy": "Welsh",
    "xh": "Xhosa",
    "yi": "Yiddish",
    "yo": "Yoruba",
    "zu": "Zulu"
}



os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"


st.header("Skillsense Web")
#sf = st.file_uploader("Choose a PDF file")
pdf = None
link = None

side = st.chat_input("What's up")
#print(pdf.name)
vertexai.init(project="speechrec-396114", location="us-central1")

llm = VertexAI(
    model_name="text-bison@001",
    max_output_tokens=512,
    temperature=0.1,
    top_p=0.8,
    top_k=40,
    verbose=True,)

embeddings = VertexAIEmbeddings()

def get_text(url):
   # Send a GET request to the URL
   response = requests.get(url)

   # Create a BeautifulSoup object with the HTML content
   soup = BeautifulSoup(response.content, "html.parser")

   # Find the specific element or elements containing the text you want to scrape
   # Here, we'll find all <p> tags and extract their text
   paragraphs = soup.find_all("p")

   # Loop through the paragraphs and print their text
   with open("temp.txt", "w", encoding='utf-8') as file:
       # Loop through the paragraphs and write their text to the file
       for paragraph in paragraphs:
           file.write(paragraph.get_text() + "\n")

@st.cache_resource
def create_langchain_index(input_text):
    print("--indexing---")
    get_text(input_text)
    loader = TextLoader("temp.txt", encoding='utf-8')
    # data = loader.load()

    index = VectorstoreIndexCreator(vectorstore_cls=DocArrayInMemorySearch,
              embedding=embeddings).from_loaders([loader])
    # using vertex ai embeddings initialized above
    return index

def pdf_to_text(path):
    PROJECT_ID = "speechrec-396114"
    LOCATION = "eu"  
    PROCESSOR_ID = "1767332b13998984" 

    FILE_PATH = path
    MIME_TYPE = "application/pdf"

    docai_client = documentai.DocumentProcessorServiceClient(
    client_options=ClientOptions(api_endpoint=f"{LOCATION}-documentai.googleapis.com"))

    RESOURCE_NAME = docai_client.processor_path(PROJECT_ID, LOCATION, PROCESSOR_ID)

    with open(FILE_PATH, "rb") as image:
        image_content = image.read()
    
    raw_document = documentai.RawDocument(content=image_content, mime_type=MIME_TYPE)
    request = documentai.ProcessRequest(name=RESOURCE_NAME, raw_document=raw_document)
    result = docai_client.process_document(request=request)
    document_object = result.document

    with open("temp.txt","w",encoding="utf-8") as f:
        f.write(document_object.text)

def text_to_wav(voice_name: str, text: str):
    language_code = "-".join(voice_name.split("-")[:2])
    text_input = tts.SynthesisInput(text=text)
    voice_params = tts.VoiceSelectionParams(
        language_code=language_code, name=voice_name
    )
    audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.LINEAR16)

    client = tts.TextToSpeechClient()
    response = client.synthesize_speech(
        input=text_input,
        voice=voice_params,
        audio_config=audio_config,
    )

    filename = f"{voice_name}.wav"
    with open(filename, "wb") as out:
        out.write(response.audio_content)
        print(f'Generated speech saved to "{filename}"')

@st.cache_resource
def create_langchain_index_pdf(path):
    print("--indexing---")
    pdf_to_text(path)
    loader = TextLoader("temp.txt", encoding='utf-8')
    # data = loader.load()

    index = VectorstoreIndexCreator(vectorstore_cls=DocArrayInMemorySearch,
              embedding=embeddings).from_loaders([loader])
    # using vertex ai embeddings initialized above
    return index

@st.cache_resource
def train_youtube():
    print("--indexing---")
    #pdf_to_text(path)
    loader = TextLoader("temp.txt", encoding='utf-8')
    # data = loader.load()

    index = VectorstoreIndexCreator(vectorstore_cls=DocArrayInMemorySearch,
              embedding=embeddings).from_loaders([loader])
    # using vertex ai embeddings initialized above
    return index

@st.cache_data
def get_response(input_text,query):
    print(f"--querying---{query}")
    response = index.query(query,llm=llm)
    return response

def translating(text,lang):
    translate_client = translate.Client()
    #text = ans
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    get = list(supported_languages.values()).index(lang)
    value = list(supported_languages.keys())[get]
    result = translate_client.translate(text, target_language=value)

    return result["translatedText"]
#chat  = st.chat_input("Type Here")

summary_response = ""
tweet_response = ""
ln_response = ""

s = st.sidebar.radio("Choose an Option",options=['link','pdf','youtube'])


if s=='link':
    input_text=st.sidebar.text_input("Provide the link to the webpage...")
    lang = st.sidebar.selectbox(label="Choose Language",options=supported_languages.values(),index=15)
    submit = st.sidebar.button("Submit")
    if submit:
        index = create_langchain_index(input_text)
        #pod = st.sidebar.button("Generate Podcast !!")
        summary_query ="Write a 500 words summary of the document"
        summary_response = get_response(input_text,summary_query)

        summary_response_ta = translating(summary_response,"Tamil")
        print(summary_response_ta)
        text_to_wav("en-IN-Wavenet-C", summary_response) 
        text_to_wav("ta-IN-Wavenet-C", summary_response_ta) 

        #st.sidebar.write("Podcast")

        #audio_file = open('K:\PsgHack\Langchain-streamlit\\ta-IN-Wavenet-C.wav', 'rb')
        #audio_bytes = audio_file.read()

        #st.sidebar.audio(audio_bytes, format='audio/wav')
    
        #st.audio("")
    
        #response = get_response(query=chat)
if s=='pdf':
    pdf = st.sidebar.file_uploader("Choose a PDF File")
    lang = st.sidebar.selectbox(label="Choose Language",options=supported_languages.values(),index=15)
    if pdf:
        with open(os.path.join("files",pdf.name),"wb") as f:
            f.write(pdf.getbuffer())

        path = f"files\{pdf.name}"
        submit = st.sidebar.button("Submit")
        if submit:
            index = create_langchain_index_pdf(path=path)
            #pod = st.sidebar.button("Generate Podcast !!")
            summary_query ="Write a 500 words summary of the document"
            summary_response = get_response(pdf,summary_query)

            summary_response_ta = translating(summary_response,"Tamil")
            print(summary_response_ta)
            text_to_wav("en-IN-Wavenet-C", summary_response) 
            text_to_wav("ta-IN-Wavenet-C", summary_response_ta)

if s=='youtube':
    input_link=st.sidebar.text_input("Provide the link to the webpage...")
    lang = st.sidebar.selectbox(label="Choose Language",options=supported_languages.values(),index=15)
    submit = st.sidebar.button("Submit")
    print(input_link)
    if submit:
        print("Proceeding")
        link = input_link.split('=')[-1]
        outls = []
        tx = YouTubeTranscriptApi.get_transcript(link)
        for i in tx:
            o = i['text']
            outls.append(o)
        print(outls)
        with open("temp.txt","w",encoding="utf-8") as f:
            f.writelines(outls)
        print("Done")

        index = train_youtube()
        summary_query ="Write a 500 words summary of the document"
        summary_response = get_response(link,summary_query)

        summary_response_ta = translating(summary_response,"Tamil")
        print(summary_response_ta)
        text_to_wav("en-IN-Wavenet-C", summary_response) 
        text_to_wav("ta-IN-Wavenet-C", summary_response_ta)



    
        #summary_query ="Write a 100 words summary of the document"
        #summary_response = get_response(input_text=None,query= summary_query)
st.sidebar.title("Smart Pod 🎧")
n = st.sidebar.selectbox(label="Select Language",options=['Tamil','English'])
co = st.sidebar.button("Submit",key=2)
if co:
    if n=="Tamil":
        audio_file = open('K:\PsgHack\Langchain-streamlit\\ta-IN-Wavenet-C.wav', 'rb')
        audio_bytes = audio_file.read()
        st.sidebar.audio(audio_bytes, format='audio/wav')
    if n=="English":
        audio_file = open('K:\PsgHack\Langchain-streamlit\en-IN-Wavenet-C.wav', 'rb')
        audio_bytes = audio_file.read()
        st.sidebar.audio(audio_bytes, format='audio/wav')

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if query := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(query)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    #query = "about?"
    #docs = db.similarity_search(query)
    if s=='pdf':
        index = create_langchain_index_pdf(path=path)
    if s=='link':
        index = create_langchain_index(input_text)
    if s=='youtube':
        index = train_youtube()
    ans = get_response(input_text=None,query=query)


    result = translating(ans,lang)
    response = f"Bluesense: {result}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": result})

