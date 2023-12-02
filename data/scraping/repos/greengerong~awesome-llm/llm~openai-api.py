from fastapi import FastAPI, Query
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI

# uvicorn llm.openai-api:app --reload
# http://127.0.0.1:8000/chat/openai?prompt=中国首都在哪

app = FastAPI()

@app.get("/chat/openai")
def ask(prompt: str = Query(..., max_length=40)):
   
    chat = ChatOpenAI()
    messages = [
        SystemMessage(content="简单回答，使用中文回复"),
        HumanMessage(content=prompt)
    ]
    res = chat(messages)
    print(res)
    return {'res': res}