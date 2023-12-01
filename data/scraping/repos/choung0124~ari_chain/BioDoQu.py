import requests
from langchain import LLMChain
from transformers import AutoTokenizer, logging
from langchain.llms import TextGen
from langchain.memory import SimpleMemory, VectorStoreRetrieverMemory, ConversationKGMemory, CombinedMemory, ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from pydantic import BaseModel
from typing import List, Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from fastapi import FastAPI
import json
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import uuid
import os
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter

app = FastAPI()

@app.middleware("http")
async def print_response(request, call_next):
    response = await call_next(request)
    print("Response status:", response.status_code)
    print("Response headers:", response.headers)
    
    if hasattr(response, 'body_iterator'):
        body = b''
        async for chunk in response.body_iterator:
            body += chunk
        print("Response body:", body.decode())
        # Rebuild the response
        response = Response(content=body, media_type=response.media_type)
    
    return response


# http://127.0.0.1:8000/api/v1/thread/document
# 
# 

model_url = "https://rose-choir-mentioned-normal.trycloudflare.com/"
llm = TextGen(model_url=model_url, max_new_tokens=512)

prompt_template = """### Instruction: 
Answer the question, using information from the context below:
Context: {input}
Question: {question}

### Response:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["question", "input"]
)
llmchain = LLMChain(llm=llm, prompt=prompt)

class Query(BaseModel):
    query: str
    doc_id: str

class Reference(BaseModel):
    title: str
    desc: str
    src: int
    link: str

class Result(BaseModel):
    response_id: str
    content: str
    refs: Optional[List[Reference]]

class Error(BaseModel):
    code: int
    message: str

class ThreadResponse(BaseModel):
    result: Optional[Result]
    error: Optional[Error]


@app.post("/api/thread/ask", response_model=ThreadResponse)
async def run_chain(item: Query):
    question = item.query
    retriever = global_faiss_index.as_retriever(search_kwargs={"k": 1})
    context = retriever.get_relevant_documents(question)
    try:
        result_content = llmchain.predict(question=question, input=context)
        # Here you should construct your references list
        refs = [
            {
                "title": "The Long-Term Denfits fo Increased Aspirin Use by At-Risk Americans Aged 50 and Older",
                "desc": "Aspirin reduces the risk of major CVD events (total MI, total stroke, CVD mortality) by 11 percent",
                "src": 1,
                "link": "https://pubmed.ncbi.nlm.nih.gov/31704172/"
            },

        ]
        return {"result": {"response_id": "UmVzcG9uc2VJRA==", "content": result_content, "refs": refs}}
    except Exception as e:
        return {"result": None, "error": {"code": 9999, "message": str(e)}}
    



@app.post("/api/v1/thread/document")
async def upload_document(file: UploadFile = File(...)):
    def generate_document_id():
        return str(uuid.uuid4())[:16]

    hf = HuggingFaceEmbeddings(
        model_name='pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    try:
        file_path = f"uploads/{file.filename}"  # Specify the path where you want to save the file
        
        # Create the 'uploads' directory if it doesn't exist
        os.makedirs("uploads", exist_ok=True)
        
        with open(file_path, "wb") as f:
            f.write(await file.read())

        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        faiss_index = FAISS.from_documents(pages, hf)
        # Generate a document id
        doc_id = generate_document_id()
        # Set the global variable
        global global_faiss_index
        global_faiss_index = faiss_index
        return {"document": {"doc_id": doc_id},}
    except Exception as e:
        # Return an error message if anything goes wrong
        return {"document": None, "error": {"code": 500, "message": str(e)}}









