import os
from typing import List, Literal, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langchain.globals import set_debug
from pydantic import BaseModel

from llm_diag import DiagnosisLLM

# set_debug(True)

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Source(BaseModel):
    source: str
    title: str


class Message(BaseModel):
    role: Literal["DOCTOR", "PATIENT", "AI"]
    content: str
    sources: Optional[List[Source]] = None


class Query(BaseModel):
    transcript: List[Message]
    chat_history: List[Message]


model = DiagnosisLLM()
model.init_extraction_chains()


def transcript_to_str(transcript: list[Message]) -> str:
    result = []
    for message in transcript:
        role = message.role.capitalize()
        result.append(f"{role}:\n{message.content}")
    return "\n\n".join(result)


@app.post("/query")
async def query_agent(query: Query):
    transcript = query.transcript
    chat_history = query.chat_history

    if len(transcript) == 0:
        raise HTTPException(status_code=400, detail="Transcript cannot be empty")

    if model.keywords is None:
        transcript_text = transcript_to_str(transcript)
        model.extract_from_transcript(transcript_text)

    # if model.knowledge is None:  # Can refetch after each round too
    #     model.init_conv_chain()
    model.init_conv_chain()

    current_message = chat_history[-1].content

    if len(chat_history) <= 1:  # Initial request
        model.memory.chat_memory.messages = []  # Clear memory

    response = model.answer_doctor_query(current_message)
    output = response["chat_history"][-1]["content"]
    sources = []
    for key in ("guidelines", "web"):
        for url in response["sources"][key]:
            sources.append(Source(title=url, source=url))
    chat_history.append(Message(role="AI", content=output, sources=sources))

    return {"response": {"chat_history": chat_history}}


app.mount("/", StaticFiles(directory="frontend/build", html=True), name="static")

if __name__ == "__main__":
    USE_NGROK = os.environ.get("USE_NGROK", False)

    if USE_NGROK:
        import nest_asyncio
        from pyngrok import ngrok

        port = 5000
        public_url = ngrok.connect(port).public_url
        nest_asyncio.apply()

        print(f"Running on {public_url}")

    uvicorn.run("frontend:app", host="0.0.0.0", port=5000, reload=True)
