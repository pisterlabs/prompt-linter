# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: python3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # infer
#
# > 

# %%
# | default_exp infer

# %%
# | hide
from nbdev.showdoc import *
from emblem.api import OpenAIModel
from emblem.data import Chunks

# %%
# | export
from typing import Union
from typing import Callable
from fastapi import FastAPI
from emblem.core import EmbeddingModel
from emblem.core import EmbeddingConfig
from pydantic import BaseModel


# %%
#| export
class ChunkInput(BaseModel):
    path: str


# %%
#| export
class ChunkOutput(BaseModel):
    path: str


# %%
#| export
class ModelInput(BaseModel):
    text: str


# %%
#| export
class ModelOutput(BaseModel):
    embedding: list[float]


# %%
# | export
def server(
        models: dict[str, EmbeddingModel],
        chunkers: dict[str, Callable]
    ) -> None:
    app = FastAPI()
    
    routes_list = ""
    for model_name in models.keys():
        routes_list += f"/model/{model_name} "

    for chunk_name in chunkers.keys():
        routes_list += f"/chunk/{chunk_name} "

    for model_name, model in models.items():
        @app.post(f"/model/{model_name}")
        async def get_model_prediction(model_input: ModelInput) -> ModelOutput:
            prediction = model.predict(model_input.text)
            return ModelOutput(prediction=prediction)

    for chunk_name, chunker in chunkers.items():
        @app.post(f"/chunk/{chunk_name}")
        async def chunk_document(chunk_input: ChunkInput) -> ChunkOutput:
            output_path = chunker(path=chunk_input.path)
            return ModelOutput(path=output_path)

    @app.get("/{route}")
    def catch_all(route: str):
        return {f"Invalid route /{route}, please call one of: {str.trim(routes_list)}"}

    routes = [{"path": route.path, "name": route.name} for route in app.routes]
    print("ROUTES: ", routes)
    uvicorn.run(app, host="0.0.0.0", port=8000)


# %%
# | hide
import nbdev

nbdev.nbdev_export()

# %%
