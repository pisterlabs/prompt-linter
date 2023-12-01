from typing import Dict

import fastapi
import fastapi.middleware.cors
import uvicorn
from pydantic import BaseModel

from server.providers.cohere import Cohere
from server.providers.huggingface import HuggingFace
from server.providers.internal_server import InternalServer
from server.providers.openai import OpenAI

OPENAI_API_KEY = "<ENTER VALUE>"
openai_provider = OpenAI(OPENAI_API_KEY)
huggingface_provider = HuggingFace()
internalserver_provider = InternalServer()
COHERE_API_KEY = "<ENTER VALUE>"
cohere_provider = Cohere(COHERE_API_KEY)

## Setup FastAPI server.
app = fastapi.FastAPI()
app.add_middleware(
    fastapi.middleware.cors.CORSMiddleware,
    allow_origins=["*", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def health_check():
    return "LLMHub is up and running."


class CompletionRequest(BaseModel):
    prompt: str
    input: Dict
    config: Dict


@app.post("/completion")
async def get_completion(request: CompletionRequest):
    # HACK
    config = request.config
    config["stopSequences"] = config["stopSequences"].replace("\\n", "\n")

    if config["engine"] == "flan-t5-xl":
        output, num_tokens, duration_s = huggingface_provider(request.prompt, request.input, config)
    elif config["engine"] == "codegen-16B-multi":
        output, num_tokens, duration_s = internalserver_provider(
            request.prompt, request.input, config
        )
    elif "cohere-" in config["engine"]:
        config["engine"] = config["engine"].replace("cohere-", "")
        output, num_tokens, duration_s = cohere_provider(request.prompt, request.input, config)
    else:
        output, num_tokens, duration_s = openai_provider(request.prompt, request.input, config)

    return_val = {"output": output, "num_tokens": num_tokens, "duration_s": duration_s}
    print(return_val)

    return return_val


@app.get("/shields/stars/{user}/{repo}")
def shields_stars(user: str, repo: str):
    print(user, repo)
    return {"schemaVersion": 1, "label": "LLMHub ⭐️", "message": "0", "color": "brightgreen"}


if __name__ == "__main__":
    # start server
    uvicorn.run(app, host="127.0.0.1", port=58001, log_level="info")
