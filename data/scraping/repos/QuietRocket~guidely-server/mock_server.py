# Configuring guidance
import guidance

guidance.llms.Mock.end_of_text = lambda x: "<|endoftext|>"

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

    llmParams = []
    if "llm" in reqVariables:
        llmParams = reqVariables["llm"]
        del reqVariables["llm"]

    async def event_stream():
        previous = ""

        program = guidance(reqProgram, llm=guidance.llms.Mock(llmParams))
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
