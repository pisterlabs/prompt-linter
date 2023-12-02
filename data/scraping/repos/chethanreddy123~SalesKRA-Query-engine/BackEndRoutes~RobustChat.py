from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import (ConversationBufferMemory, 
                                                  ConversationSummaryMemory, 
                                                  ConversationBufferWindowMemory,
                                                  ConversationKGMemory)
from langchain.callbacks import get_openai_callback
from langchain.llms import GooglePalm
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import json
from pymongo.mongo_client import MongoClient
from fastapi.middleware.cors import CORSMiddleware
import logging

llm = GooglePalm(
    model='models/text-bison-001',  
    temperature=0,
    # The maximum length of the response
    max_output_tokens=80000,
    google_api_key='AIzaSyA1fu-ob27CzsJozdr6pHd96t5ziaD87wM'
)

MongoDB_Key = "mongodb://aioverflow:12345@ac-pu6wews-shard-00-00.me4dkct.mongodb.net:27017,ac-pu6wews-shard-00-01.me4dkct.mongodb.net:27017,ac-pu6wews-shard-00-02.me4dkct.mongodb.net:27017/?ssl=true&replicaSet=atlas-jcoztp-shard-0&authSource=admin&retryWrites=true&w=majority"
Data = MongoClient(MongoDB_Key)
EmployeeData = Data['FinalAxisBankHackathon']['EmployeeData']
KRAsData = Data['FinalAxisBankHackathon']['KRAsData']

def count_tokens(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(f'Spent a total of {cb.total_tokens} tokens')

    return result

def load_json_file(file_path):
    with open(file_path, 'r') as json_file:
        json_string = json_file.read()
    return json_string

documents = EmployeeData.find()

# Convert documents to JSON strings
json_string = [json.dumps(doc, default=str) for doc in documents]



# Initialize your LLM chain and conversation chain
llm = GooglePalm(
    model='models/text-bison-001',
    temperature=0,
    max_output_tokens=80000,
    google_api_key='AIzaSyA1fu-ob27CzsJozdr6pHd96t5ziaD87wM'
)

conversation_kg = ConversationChain(
    llm=llm, 
    memory=ConversationKGMemory(llm=llm)
)


# Initialize conversation_kg with the initial message
initial_message = f'''Good morning AI!, You are an Expert 
in Sales Key Result Areas (KRA) Setting and Performance Management. 
You are here to help me with my queries regarding Sales Key Result Areas 
(KRA) Setting and Performance Management, And all the sales employee data is 
given below for future analysis: {json_string}'''

logging.info(initial_message)


check = conversation_kg(initial_message)


# Initialize FastAPI app
app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request and response models (if needed)
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str

# Define API endpoint for chat
@app.post("/chat/")
async def chat(request: Request):
    json_data = await request.json()
    query = json_data.get("query")

    try:
        response = count_tokens(conversation_kg, query)
        return {"response": response}
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))

# Define a route for initial setup (if needed)
@app.get("/")
def initial_setup():
    return {"message": "Server is up and running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
