from typing import Any, Dict, Generator, List

from fastapi import FastAPI
from pydantic import BaseModel

import openai

openai.api_key = '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'

app = FastAPI()

class Message(BaseModel):
    role: str
    content: str

def stream_chat_responses(response: Any) -> Generator[Dict[str, str], None, None]:
    """
    Handles streaming responses from the OpenAI API.
    Formats the response as a dictionary with 'role' and 'content' keys.
    """
    role = None
    content = ""
    for chunk in response:
        delta = chunk['choices'][0]['delta']
        if 'role' in delta:
            # If there's a role in this chunk, yield any previous message
            if role is not None:
                yield {"role": role, "content": content.strip()}
            role = delta['role']
            content = ""
        elif 'content' in delta:
            # If there's content in this chunk, add it to the content string
            content += delta['content']
    # Yield the last message
    if role is not None:
        yield {"role": role, "content": content.strip()}

@app.post("/chat")
async def chat(history: List[Message]):
    #turn history into a list of dicts
    history = [msg.dict() for msg in history]

    # Append a system message to the history
    history.append({"role": "system", "content": "You are a helpful AI!"})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=history,
        temperature=0.5,
        max_tokens=50,
        stream=True
    )

    return list(stream_chat_responses(response))
