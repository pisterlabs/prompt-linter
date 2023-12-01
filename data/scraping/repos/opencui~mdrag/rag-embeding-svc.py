#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import requests
from urllib.parse import urljoin

from aiohttp import web
from typing import Any, List
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.base import BaseEmbedding

langchain_embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1",
    model_kwargs={'device': 'cpu'},
)

routes = web.RouteTableDef()


@routes.get("/")
async def hello(_: web.Request):
    return web.Response(text="Hello, world")


# curl -v -d '["doc1", "doc2"]' 127.0.0.1:8080/encode
@routes.post("/encode")
async def encode(req: web.Request):
    _body = await req.read()
    try:
        body = json.loads(_body)
        if type(body) != list:
            return web.Response(status=500, text="post body must be [] object")
    except Exception as e:
        return web.Response(status=500, text=str(e))

    resp = langchain_embedding.embed_documents(body)
    return web.json_response(data=resp)


class RemoteEmbeddings(BaseEmbedding):

    def __init__(self, embedding_svc: str = "", **kwargs: Any) -> None:
        if embedding_svc == "":
            raise

        self.embedding_svc = urljoin(embedding_svc, "/")
        super().__init__(**kwargs)

    def http_post(self, body: List[str]) -> Any:
        url = f'${self.embedding_svc}encode'
        data = requests.post(url, json=json.dumps(body)).json()
        return data

    def _get_query_embedding(self, query: str) -> List[float]:
        data = self.http_post(body=[query])
        return data

    def _get_text_embedding(self, text: str) -> List[float]:
        data = self.http_post(body=[text])
        return data

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        data = self.http_post(body=texts)
        return data


if __name__ == "__main__":
    app = web.Application()
    app.add_routes(routes)
    web.run_app(app)
