# app.py

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from collections import defaultdict
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from fetch_llm import hf_model

load_dotenv()

app = FastAPI(title="ConversationalRetrievalChainDemo")
templates = Jinja2Templates(directory="templates")




# ################uncomment the following code if you are using a seperate frontend code#########
# Configure CORS settings

# from fastapi.middleware.cors import CORSMiddleware

# origins = [
#     "http://localhost:5173",  # Update this to your frontend URL
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )





##if using hugging face
llm = hf_model()



# if using replicate
# llm = replicate_llm()



def create_chain():
    from langchain.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings()

    from langchain.vectorstores import Chroma
    docsearch = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True
    )


chain = create_chain()
chat_history = defaultdict(list)


class ChatRequest(BaseModel):
    Userid: str
    query: str


@app.post("/Chat_me")
def chat_me(request: ChatRequest):
    Userid = request.Userid
    query = request.query
    result = chain({'question': query, 'chat_history': chat_history[Userid]})
    chat_history[Userid].append((query, result['answer']))
    file1 = open("{0}.txt".format(Userid), "a")  # append mode
    file1.write(query + " " + result['answer'] + "\n")
    file1.close()
    return {"response": 'Answer: ' + result['answer']}


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
