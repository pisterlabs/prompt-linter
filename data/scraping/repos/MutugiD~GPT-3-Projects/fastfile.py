import os
from typing import List
import uvicorn
from fastapi import FastAPI
from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI

os.environ["OPENAI_API_KEY"] = ""

app = FastAPI()

def create_index(path):
    max_input = 4096
    tokens = 200
    chunk_size = 600 #for LLM, we need to define chunk size
    max_chunk_overlap = 20

    #define prompt
    promptHelper = PromptHelper(max_input,tokens,max_chunk_overlap,chunk_size_limit=chunk_size)

    #define LLM — there could be many models we can use, but in this example, let’s go with OpenAI model
    llmPredictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-ada-001",max_tokens=tokens))

    #load data — it will take all the .txtx files, if there are more than 1
    docs = SimpleDirectoryReader(path).load_data()

    #create vector index
    vectorindex = GPTSimpleVectorIndex(documents=docs,llm_predictor=llmPredictor,prompt_helper=promptHelper)
    vectorindex.save_to_disk("vectorIndex.json")
    return "vectorIndex.json"

def answerMe(prompt, vectorindex):
    vIndex = GPTSimpleVectorIndex.load_from_disk(vectorindex)
    response = vIndex.query(prompt,response_mode="compact")
    return response

@app.get("/")
async def root():
    return {"message": "Welcome to Diani App, ask me any questions about hotels in Diani."}

@app.post("/answer")
async def answer(question: str):
    # Get response from the bot
    response = answerMe(question, "vectorIndex.json")
    return {"answer": response}

if __name__ == "__main__":
    # The path where the documents for the bot are stored
    path = "Knowledge"
    # Create vector index and save to disk
    vectorindex_path = create_index(path)
    # Run the app
    uvicorn.run(app, host="0.0.0.0", port=8000)
