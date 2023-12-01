# Setting up the model
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

model_name_or_path = "TheBloke/Llama-2-13B-chat-GPTQ"
model_basename = "gptq_model-4bit-128g"

use_triton = True

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

model = AutoGPTQForCausalLM.from_quantized(
    model_name_or_path,
    revision="gptq-4bit-32g-actorder_True",
    model_basename=model_basename,
    use_safetensors=True,
    trust_remote_code=True,
    device="cuda:0",
    quantize_config=None,
    use_triton=use_triton,
)

# Configuring guidance
import guidance

guidance.llm = guidance.llms.Transformers(
    model=model, tokenizer=tokenizer, device="cuda:0"
)

# Text diffing function
import difflib


def diff_strings(string1, string2):
    diff = difflib.ndiff(string1, string2)
    diff_list = list(diff)
    difference = "".join([i[2:] for i in diff_list if i[0] != " "])
    return difference


# Configuring the server
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse, PlainTextResponse
from starlette.routing import Route
from starlette.requests import Request


async def llm(request: Request):
    body = await request.json()

    if "program" not in body:
        return PlainTextResponse("Program not specified", status_code=400)
    if "variables" not in body:
        return PlainTextResponse("Variables not specified", status_code=400)

    reqProgram = body["program"]
    reqVariables = body["variables"]

    async def event_stream():
        previous = ""
        program = guidance(reqProgram)
        async for p in program(
            stream=True, async_mode=True, silent=True, **reqVariables
        ):
            diff = diff_strings(previous, p.text)
            yield diff
            previous = p.text

    return StreamingResponse(event_stream(), media_type="text/event-stream")


routes = [
    Route("/llm", llm, methods=["POST"]),
]

app = Starlette(debug=True, routes=routes)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
