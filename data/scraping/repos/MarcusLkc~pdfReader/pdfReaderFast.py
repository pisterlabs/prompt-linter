from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import openai
import os
import time

app = FastAPI()

# Global variable to store the assistant ID
global_assistant_id = None

global_client = None

# Define the request model for the API
class QuestionRequest(BaseModel):
    question: str

# Mount the `static` directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Redirect to index.html for the root path
@app.get("/")
async def read_index():
    return FileResponse('static/index.html')


# Initialize the OpenAI client
def get_openai_client():
    # check if global_client is none if so create a new client
    global global_client
    if global_client is None:
        global_client = create_openai_client()
    return global_client

def create_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("The OPENAI_API_KEY environment variable is not set.")
    return openai.OpenAI(api_key=api_key)

# Function to create an assistant with a given file
async def create_assistant(file: UploadFile, client):
    global global_assistant_id
    try:
        contents = await file.read()
        response = client.files.create(file=contents, purpose='assistants')
        file_id = response.id

        assistant = client.beta.assistants.create(
            name="Custom Real Estate Advisor",
            instructions="You are smart assistant. Who can read any pdf or document. Use documents as firsy source of truth",
            tools=[{"type": "retrieval"}],
            model="gpt-4-1106-preview",
            file_ids=[file_id]
        )
        global_assistant_id = assistant.id
        return global_assistant_id
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Function to ask a question to the assistant
def ask_question(question, client):
    global global_assistant_id
    if global_assistant_id is None:
        raise HTTPException(status_code=400, detail="Assistant has not been created yet.")
    try:
        # Create a thread
        thread = client.beta.threads.create()
        thread_id = thread.id

        # Send the user question to the assistant
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=question
        )

        # Retrieve the response
        run = client.beta.threads.runs.create(thread_id=thread_id, assistant_id=global_assistant_id)
        run_id = run.id

        # Poll for the completion of the run
        run_status = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
        while run_status.status != "completed":
            # Implement an appropriate wait strategy
            time.sleep(2)
            run_status = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)

        # Retrieve the last message from the assistant
        messages = client.beta.threads.messages.list(thread_id=thread_id)
        last_message_for_run = [
            message for message in messages.data if message.run_id == run_id and message.role == 'assistant'
        ][-1]

        return last_message_for_run.content[0].text.value
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to upload a file and create an assistant
@app.post("/upload_and_create_assistant")
async def upload_and_create_assistant(file: UploadFile = File(...)):
    client = get_openai_client()
    assistant_id = await create_assistant(file, client)
    return {"assistant_id": assistant_id}

# Endpoint to ask a question without needing to provide an assistant ID
@app.post("/ask")
async def ask_endpoint(question_request: QuestionRequest):
    client = get_openai_client()
    response = ask_question(question_request.question, client)
    print('response', response)
    return {"answer": response}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
