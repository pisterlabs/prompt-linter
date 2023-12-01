from collections import defaultdict
import cohere

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder
from config import QdrantConfig, CohereConfig, ModelsConfig
from utils import (
    build_prompt,
    clean_question,
    get_similar_docs,
    rerank_docs,
    get_response,
)

cohere_client = cohere.Client(api_key=CohereConfig.api_key)
qdrant_client = QdrantClient(
    host=QdrantConfig.host,
    port=QdrantConfig.port,
    api_key=QdrantConfig.api_key,
)
reranking_model = CrossEncoder(ModelsConfig.reranking_model_name)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/ask", response_class=HTMLResponse)
def ask(
    request: Request,
    question: str = Form(),
):
    response = "Sorry, I don't know the answer."

    try:
        question_cl = clean_question(question)
    except ValueError:
        return HTMLResponse(
            content=response,
            status_code=200,
        )

    hits = get_similar_docs(
        query=question_cl,
        qdrant_client=qdrant_client,
        cohere_client=cohere_client,
        collection_name=QdrantConfig.collection_name,
    )
    reranked_results = rerank_docs(hits=hits, query=question_cl, model=reranking_model)
    prompt = build_prompt(question_cl, reranked_results)

    response = get_response(prompt, cohere_client=cohere_client)

    references = defaultdict(list)
    for result in reranked_results:
        references[result["title"]].append(result)

    references = defaultdict(
        list, {k: v for k, v in references.items() if any([r["section"] for r in v])}
    )

    return templates.TemplateResponse(
        "_response.html",
        {
            "request": request,
            "generated_text": response.strip().split("\n"),
            "references": references,
        },
    )
