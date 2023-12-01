"""
Processors for different models
"""
import functools
import json
import logging
from typing import List

import cohere as co
import httpx
import openai
import tiktoken

from emb3d import config
from emb3d.types import Backend, EmbedJob, EmbedResponse, Failure, Result, WaitFor

HF_HEADERS = {}
OPENAI_INIT_PARAMS = {}
COHERE_INIT_PARAMS = {}

OPENAI_ENDPOINT = "https://api.openai.com/v1/embeddings"

_cleanup_callables = []


@functools.cache
def httpx_client() -> httpx.AsyncClient:
    """Cached httpx client"""
    cli = httpx.AsyncClient()
    _cleanup_callables.append(cli.aclose)
    return cli


@functools.cache
def cohere_client(api_key: str) -> co.AsyncClient:
    """Cached cohere client"""
    cli = co.AsyncClient(api_key=api_key)
    _cleanup_callables.append(cli.close)
    return cli


def hf_headers(job: EmbedJob) -> dict:
    """Returns the headers for the HuggingFace API"""
    return {
        "Authorization": f"Bearer {job.api_key}",
    }


async def cleanup():
    """Cleanup any resources used by the clients"""
    for cleanup_fn in _cleanup_callables:
        await cleanup_fn()


def hf_inference_url(model_id: str):
    """Inference URL for the HuggingFace API"""
    return (
        f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
    )


async def _huggingface(job: EmbedJob, inputs: List[str]) -> EmbedResponse:
    data = {"inputs": inputs, "wait_for_model": True}
    response = await httpx_client().post(
        hf_inference_url(job.model_id), headers=hf_headers(job), json=data
    )

    logging.debug("Model: %s, Response: %s", job.model_id, response.status_code)

    try:
        json_response = response.json()
    except json.JSONDecodeError:
        return Failure("HF response is not in JSON format, {response.content}")
    if response.status_code == 200:
        return Result(json_response)
    elif response.status_code == 503:
        estimated_time = json_response.get("estimated_time")
        if estimated_time:
            return WaitFor(estimated_time, None)
        else:
            return Failure(
                "[HuggingFace] Service unavailable and estimated time not provided"
            )
    return Failure(f"[HuggingFace] Unexpected response: {json_response}")


async def _openai(job: EmbedJob, inputs: List[str]) -> EmbedResponse:
    try:
        resp = await openai.Embedding.acreate(model=job.model_id, input=inputs)
        return Result([row.embedding for row in resp.data])
    except openai.error.RateLimitError as err:
        logging.debug("[OpenAI] Rate limit error: %s", err)
        return WaitFor(config.RATE_LIMIT_WAIT_TIME_SECS, error=err)
    except openai.error.APIError as err:
        logging.debug("[OpenAI] Error:", err)
        return Failure(f"[OpenAI] Error: {err}")


async def _cohere(job: EmbedJob, inputs: List[str]) -> EmbedResponse:
    cli = cohere_client(job.api_key)
    try:
        co_resp = await cli.embed(inputs, job.model_id)
        return Result(co_resp.embeddings)
    except co.error.CohereError as err:
        return Failure(f"[Cohere] Error: {err}")


async def gen(job: EmbedJob, inputs: List[str]) -> EmbedResponse:
    """
    Generate embeddings for the given inputs.

    Routes to the appropriate client based on the job specification.
    """
    if job.backend == Backend.HUGGINGFACE:
        return await _huggingface(job, inputs)
    elif job.backend == Backend.OPENAI:
        openai.api_key = job.api_key
        return await _openai(job, inputs)
    elif job.backend == Backend.COHERE:
        return await _cohere(job, inputs)
    else:
        raise ValueError(f"Unknown backend: {job.backend}")


@functools.cache
def get_encoder(model_id: str) -> tiktoken.Encoding:
    """Returns encoder used for the given model_id"""
    return tiktoken.encoding_for_model(model_id)


def approx_token_count(job: EmbedJob, input: str) -> int:
    """
    Helper to compute the approximate token count for an input

    For backends other than OpenAI, we use an approximate method as a fast
    tokenizer isn't always available.
    """
    if job.backend == Backend.OPENAI:
        encoder = get_encoder(job.model_id)
        return len(encoder.encode(input))
    return len(input) // 2
