import asyncio
import json
import traceback
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from langchain.callbacks.base import CallbackManager, AsyncCallbackHandler, AsyncCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from rag_with_cog_search import rag

BEGIN_TOKEN = "<|begin|>"
END_TOKEN = "<|end|>"
ERR_TOKEN = "<|err|>"
app = FastAPI()


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    streaming_handler = StreamingLLMCallbackHandler(websocket)
    streaming_manager = AsyncCallbackManager([streaming_handler])
    while True:
        try:
            req = await websocket.receive_text()
            resp = {"token": BEGIN_TOKEN}
            await websocket.send_json(resp)

            req: dict = json.loads(req)
            result = await rag(req.get("question"), top=req.get("top", 3), streaming_callback_manager=streaming_manager)
            print("Q:", result["query"])
            print("A:", result["result"][:50] + "...")

            resp = {"token": END_TOKEN}
            await websocket.send_json(resp)
        except WebSocketDisconnect:
            print("websocket disconnect")
            break
        except Exception as e:
            print(f"Exception occurred: {e}")
            traceback.print_exc()            
            await websocket.send_json({"token": ERR_TOKEN})


class StreamingLLMCallbackHandler(AsyncCallbackHandler):
    """Callback handler for streaming LLM responses."""

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        print(token, end="")
        resp = {"token": token}
        await self.websocket.send_json(resp)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9999)
