from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import LLMChain
import uvicorn
from prompts import QA_PROMPT
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from utils import ChatResponse, StreamingLLMCallbackHandler, get_or_create_chatgroup_vector_db
from langchain.chains.summarize import load_summarize_chain
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.schema import (
    HumanMessage,
)

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    stream_handler = StreamingLLMCallbackHandler(websocket)
    handler = StreamingStdOutCallbackHandler()

    OPENAI_API_KEY_SUMMARY = 'sk-Icpn09CQJ5Pp6' 
    summaryllm = ChatOpenAI(temperature=0.5, openai_api_key=OPENAI_API_KEY_SUMMARY)

    OPENAI_API_KEY_EMBEDDING = 'sk-'
    embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY_EMBEDDING)

    OPENAI_API_KEY_ANSWER = "sk-Qkgetd"

    chatllm = ChatOpenAI(temperature=0.8, openai_api_key=OPENAI_API_KEY_ANSWER, streaming=True, verbose=True)

    while True:
        await websocket.send_json({"type": "start"})
        question = await websocket.receive_text()
        # resp = ChatResponse(sender="you", message=question, type="stream")
        # await websocket.send_json(resp.dict())

        # # Construct a response
        # start_resp = ChatResponse(sender="bot", message="", type="start")
        # await websocket.send_json(start_resp.dict())

        # db = get_or_create_chatgroup_vector_db("chat_id", embedding, "store")
  
        # print("正在读取与提问相关联的记忆...")  
        # docs = db.similarity_search(query=question, k=4)
        # # refine question
        # chain = load_summarize_chain(summaryllm, chain_type="stuff")
        # summary = chain.run(docs)
        # print("总结上下文如下：", summary)


        # format_question = QA_PROMPT.format_prompt(context=summary, question=question).to_string()
        await chatllm.apredict(question)

        # print("正在添加聊天记录至记忆库...", ["Human: " + question, "Assistant: " + res])
        # db.add_texts(["Human" + question, "Assistant: " + res])
        # db.persist()

        # end_resp = ChatResponse(sender="bot", message="", type="end")
        await websocket.send_json({"type": "end"})

        

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)