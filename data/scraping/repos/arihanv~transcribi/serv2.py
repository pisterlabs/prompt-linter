from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import OpenAI
from langchain import PromptTemplate
# from trapy import Trapy
import json
from trapy import Trapy
import time
import asyncio
import os
import app_secrets


origins = [
    "http://localhost:3000",
    "localhost:3000",
    "http://localhost:3001",
    "localhost:3001",
]

trapy = Trapy()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

transcript = "test"
db = None
llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=app_secrets.OPEN_AI_KEY)


@app.get("/")
def read_root():
    b = trapy.load_transcript('transcript.json')['segments']
    tran = trapy.format_transcript()
    chunks = trapy.chunker(tran, 3)
    return chunks, tran, b

class Message(BaseModel):
    message: str
    transcript: str

@app.post("/analyze")
def analyze(data: Message):
    global db
    global llm
    docs = db.similarity_search(data.message, k=2)
    template = """
    Answer the following question based on the video transcript:
    Question:{question}
    Transcript:{transcript}
    """
    prompt = PromptTemplate(
    input_variables=["transcript", "question"], 
    template=template
    )
    qa_prompt = prompt.format(transcript=docs, question=data.message)
    res = llm(qa_prompt)
    return res
    # print(qa_prompt)
    # return "hello"

@app.post("/index")
def index(data: Message):
    global transcript
    global db
    transcript = data.transcript
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_text(transcript)
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    db = Chroma.from_texts(docs, embeddings)
    return {"message": "success"}

@app.websocket("/wss")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        print('sd')
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        await websocket.close()

@app.websocket("/wst")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        split_data = []
        prev_dur = 0
        should_stop = False

        await asyncio.sleep(0)

        def num_splits(folder_path):
            file_list = os.listdir(folder_path)
            file_list = [file for file in file_list if os.path.isfile(os.path.join(folder_path, file))]
            file_count = len(file_list)
            return file_count

        # num_splits('splits') + 1
        # time.sleep(1)
        await websocket.send_json({"chunk": "Loading", "segment": "Loading"})

        # with open('test.json', 'r') as f:
        #     transcript = json.load(f)
        
        for i in range(1, num_splits('splits') + 1):
            if should_stop:
                break

            dur = trapy.get_duration(f'splits/split-{i}.wav')
            if dur > 1.2:
                temp = trapy.generate(f"splits/split-{i}.wav")
                for segment in temp['segments']:
                    segment['start'] += prev_dur
                    segment['end'] += prev_dur
                    for word in segment['words']:
                        word['start'] += prev_dur
                        word['end'] += prev_dur 
                for chunk in trapy.chunker(trapy.format_transcript(temp), 3):
                    split_data.append(chunk)
                    # with open('test.json', 'w') as f:
                    #     json.dump({"chunk":chunk, "segment": temp["segments"]}, f)
                    await websocket.send_json({"chunk":chunk, "segment": temp["segments"]})
                    await asyncio.sleep(0)
                await asyncio.sleep(0)
            prev_dur += dur
            if temp['text'] == "Stop":
                should_stop = True

        # await websocket.send_json({"chunk": transcript['chunk'], "segment": transcript['segment']})

        

        await websocket.send_json({"chunk": "Done", "segment": "Done"})

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000, ws_ping_timeout=10000)