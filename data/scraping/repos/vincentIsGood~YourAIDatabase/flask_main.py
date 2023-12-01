"""
Only 1 job can run at a time, since we have 1 LLM loaded in memory only.

TODO: organize this pile of * please.
"""

import sys
import os
import json
import threading
import asyncio
from typing import Any, Dict, List, Optional
from uuid import UUID
from langchain.schema.output import LLMResult
from transformers.models.auto import AutoTokenizer
from langchain.callbacks.base import BaseCallbackHandler
from websockets.server import WebSocketServerProtocol, serve
from flask import Flask, request, send_from_directory, Response
from flask_cors import CORS
from markupsafe import escape, Markup
from transformers import TextStreamer

import configs.common as config
import adddata
import lib.utils.url_utils as url_utils
from lib.AiDatabase import AiDatabase
from lib.utils.async_utils import run_async
from lib.utils.randutils import randomString

FLASK_PORT = 5022
WEBSOCKET_PORT = 5023
UPLOAD_FOLDER = config.DOCS_DIRECTORY

app = Flask(__name__, template_folder="public", static_folder="public")
app.config['SECRET_KEY'] = "asdasdwefdgdfcvbnm,nadsjkh"
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["WEB_UPLOAD_SECRET"] = config.WEB_UPLOAD_SECRET
cors = CORS(app)
currentJob = None
queryJob = None
mainWebSocket = None

class WsTextStreamer(TextStreamer):
    """For transformers / HuggingFace LLM
    """
    def __init__(self, tokenizer: AutoTokenizer, skip_prompt: bool = False, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.buffer: 'list[str]' = []

    def on_finalized_text(self, text: str, stream_end: bool = False):
        super().on_finalized_text(text, stream_end)
        if text == "":
            return
        async def wrapper():
            if not mainWebSocket:
                self.buffer.append(text)
                if stream_end:
                    self.buffer = []
                return
            
            if len(self.buffer) > 0:
                await mainWebSocket.send(self.buffer)
                self.buffer = []
            await mainWebSocket.send(text)
        run_async(wrapper)

class StreamingCallbackHandler(BaseCallbackHandler):
    """For LangChain / CTransformers LLM
    """
    buffer: 'list[str]' = []

    def on_llm_start(self, 
                     serialized: Dict[str, Any], 
                     prompts: List[str], *, 
                     run_id: UUID, 
                     parent_run_id: 'UUID | None' = None, 
                     tags: 'List[str] | None' = None, 
                     metadata: 'Dict[str, Any] | None' = None, 
                     **kwargs: Any) -> Any:
        for prompt in prompts:
            print(prompt)
        pass

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        sys.stdout.write(token)
        sys.stdout.flush()
        async def wrapper():
            if not mainWebSocket:
                self.buffer.append(token)
                return
            
            if len(self.buffer) > 0:
                await mainWebSocket.send(self.buffer)
                self.buffer = []
            await mainWebSocket.send(token)
        run_async(wrapper)
    
    def on_llm_end(self, 
                   response: LLMResult, *, 
                   run_id: UUID, parent_run_id: 'UUID | None' = None, 
                   **kwargs: Any) -> Any:
        self.buffer = []

async def queryAndSendSources(query: str):
    global mainWebSocket, currentJob
    sources = aiDatabase.query(Markup(query).unescape())

    ## After AI response, send sources and reset
    if mainWebSocket:
        await mainWebSocket.send("[[[SOURCES]]]")
        for source in sources:
            await mainWebSocket.send(json.dumps(source.metadata))
        await mainWebSocket.send("[[[END]]]")
        mainWebSocket = None
    currentJob = None

aiDatabase = AiDatabase([StreamingCallbackHandler()], WsTextStreamer)

@app.route("/app/")
def index():
    return send_from_directory("public", "index.html")

@app.route("/app/<path:path>")
def appFiles(path):
    return send_from_directory("public", path)

@app.route("/app/docs/<path:path>")
def serveImportedDocs(path):
    return send_from_directory("docs/imported", path)

@app.route("/aidb/urlupload", methods=["POST"])
def uploadDocUrl():
    if not url_utils.isUriValid(request.data):
        return Response(status=400)
    
    loadedDocs = adddata.loadWebData([request.data])
    if len(loadedDocs) == 0:
        return Response(status=204)
    
    aiDatabase.addDocsToDb(loadedDocs)
    return Response(status=201)

@app.route("/aidb/upload", methods=["POST"])
def uploadDocument():
    filename = request.args.get("name")
    if not request.data or filename == "" or ".." in filename:
        return Response(status=400)
    
    outFilePath = os.path.normpath(os.path.join(app.config["UPLOAD_FOLDER"], filename))
    if os.path.exists(outFilePath):
        return Response(status=204)
    
    with open(outFilePath, "wb+") as f:
        f.write(request.data)
    
    loadedDocs = adddata.loadData()
    if len(loadedDocs) == 0:
        return Response(status=204)
    
    aiDatabase.addDocsToDb(loadedDocs)
    return Response(status=201)

@app.route("/aidb/viewdocs")
def viewAllDocs():
    return aiDatabase.getAllDocs()

@app.route("/aidb/removedoc", methods=["DELETE"])
def deleteDocument():
    id = request.args.get("id")
    aiDatabase.deleteDocsFromDb([id])
    return Response(status=200)

@app.route("/aidb", methods=["GET"])
def handleDatabaseQuery():
    global currentJob, queryJob
    query = request.args.get("query")
    if query and not currentJob:
        queryJob = threading.Thread(target=lambda: asyncio.run(queryAndSendSources(query)), daemon=True)
        queryJob.start()
        currentJob = randomString(10)
        return currentJob
    return ""

@app.route("/aidb", methods=["DELETE"])
def stopGenHandler():
    if currentJob == request.args.get("id"):
        aiDatabase.stopLLM()
    return ""

async def wsHandler(websocket: WebSocketServerProtocol):
    """Query WebSocket sender
    Only ONE websocket is allowed to connect. 
    """
    global mainWebSocket
    print("[+] Client WebSocket connected: %s" % str(websocket.remote_address))
    async for message in websocket:
        msgObj = json.loads(message)
        if "id" in msgObj:
            if currentJob == msgObj["id"] and not mainWebSocket:
                mainWebSocket = websocket
                await websocket.send("[[[START]]]")
            else:
                return
    
    global queryJob
    if queryJob:
        print("[+] Waiting for LLM job to complete")
        queryJob.join()
    print("[+] Closing connection for: %s" % str(websocket.remote_address))

async def websocketMain():
    print(f"[+] Starting websocket server on 0.0.0.0:{WEBSOCKET_PORT}")
    async with serve(wsHandler, "0.0.0.0", WEBSOCKET_PORT):
        await asyncio.Future()


def flask_main():
    websocketThread = threading.Thread(target=lambda: asyncio.run(websocketMain()), daemon=True)
    websocketThread.start()

    print(f"[+] Starting flask webserver")
    app.run(port=FLASK_PORT)

def create_app():
    websocketThread = threading.Thread(target=lambda: asyncio.run(websocketMain()), daemon=True)
    websocketThread.start()

    print(f"[+] Starting waitress webserver")
    return app

if __name__ == "__main__":
    flask_main()