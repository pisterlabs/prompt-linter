import pyaudio
import websocket
import base64
import json
import time
import threading
import os

from langchain.llms import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.redis import Redis
from langchain.document_loaders import TextLoader


loader = TextLoader(os.getenv("BLOG_POST_FILE_TO_READ"))
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

p = pyaudio.PyAudio()

# starts recording
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=FRAMES_PER_BUFFER
)

print(f"Input source: {p.get_default_input_device_info()['name']}\n")

# the AssemblyAI endpoint we're going to hit
URL = "wss://api.assemblyai.com/v2/realtime/ws?sample_rate=16000"

rds = Redis.from_documents(docs, embeddings,
                           redis_url="redis://localhost:6379",
                           index_name='blog_post')
retriever = rds.as_retriever(search_kwargs=dict(k=1))
llm = OpenAI(temperature=0.5) # Can be any valid LLM
chain = load_qa_with_sources_chain(llm, chain_type="stuff")

def on_message(wsapp, message):
    if message and 'text' in message:
        result = json.loads(message)
        prompt = result['text']
        print(prompt, end="\r")
        if result['text'] and result['message_type'] == 'FinalTranscript':
            print("\n")
            print(chain({"input_documents": docs, "question": result['text']},
                  return_only_outputs=True))


def print_msg(msg):
    print("sending: " + msg)


def talk_over_ws(ws):
    def run():
        while True:
            try:
                data = stream.read(FRAMES_PER_BUFFER,
                                   exception_on_overflow=False)
                data64 = base64.b64encode(data).decode("utf-8")
                json_data = json.dumps({"audio_data":str(data64)})
                wsapp.send(json_data)
            except Exception as e:
                print(e)
                exit()
            time.sleep(0.1)
    threading.Thread(target=run).start()


wsapp = websocket.WebSocketApp("wss://api.assemblyai.com/v2/realtime/ws?sample_rate=16000",
                               on_message=on_message,
                               header={"Authorization":os.getenv("API_KEY_ASSEMBLYAI")})
wsapp.on_open = talk_over_ws
wsapp.run_forever()

