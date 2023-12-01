import asyncio
from llmodels.rag import get_generate_text, build_prompt
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain.callbacks import AsyncIteratorCallbackHandler


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def run(prompt):
    stream_callback = AsyncIteratorCallbackHandler()
    generate_text = get_generate_text(stream_callback)
    task = asyncio.create_task(wrap_done(
        generate_text.arun(prompt),
        stream_callback.done)
    )
    yield '{"choices": [{"index": 0, "delta": {"content": " end"}, "context": {"followup_questions": [], "data_points": []}, "session_state": null}]}\n'
    #await asyncio.sleep(0.1)
    batch = 5
    iter = 0
    batch_string = ''
    async for token in stream_callback.aiter():
        #print(token)
        # Use server-sent-events to stream the response
        if iter == batch:
            yield '{"choices": [{"index": 0, "delta": {"content": "'+batch_string+'"}, "context": {"followup_questions": []}, "session_state": null}]}\n'
            batch_string = ''
            iter = 0
        batch_string += token
        iter += 1
    yield '{"choices": [{"index": 0, "delta": {"content": "'+batch_string+'"}, "context": {"followup_questions": []}, "session_state": null}]}\n'

    await task
    #stream_callback.done.clear()
    #await asyncio.sleep(0.1)
    yield '{"choices": [{"index": 0, "delta": {}, "context": {"followup_questions": []}, "session_state": null}]}'
    
    # yield '{"choices": [{"index": 0, "delta": {"content": " end"}, "context": {"followup_questions": [], "data_points": []}, "session_state": null}]}\n'
    # for i in range(100):
    #     print(i)
    #     yield '{"choices": [{"index": 0, "delta": {"content": "'+str(i)+'?aa "}, "context": {"followup_questions": []}, "session_state": null}]}\n'
    # yield '{"choices": [{"index": 0, "delta": {}, "context": {"followup_questions": []}, "session_state": null}]}'

@app.post("/q")
async def chat(request: Request):
    request_json = await request.json()
    messages = request_json.get("messages", [])
    prompt = build_prompt(messages)
    return StreamingResponse(run(prompt), media_type="text/event-stream")

# Test: curl http://0.0.0.0:8000/q -X POST -d '{"messages": [{"content": "How much does it cost to study a Masters program in Sweden?"}]}' -H 'Content-Type: application/json'

from typing import Awaitable
async def wrap_done(fn: Awaitable, event: asyncio.Event):
        """Wrap an awaitable with a event to signal when it's done or an exception is raised."""
        try:
            await fn
        except Exception as e:
            # TODO: handle exception
            print(f"Caught exception: {e}")
        finally:
            # Signal the aiter to stop.
            event.set()