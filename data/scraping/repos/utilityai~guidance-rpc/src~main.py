import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from types import AsyncGeneratorType
from typing import AsyncGenerator, Optional, NoReturn, Callable, Awaitable

import grpc
import grpc_health.v1.health
from grpc.aio import ServerInterceptor
from grpc_reflection.v1alpha import reflection
import guidance
import transformers.models.llama

import guidance_pb2_grpc
from guidance_pb2 import HealthCheckResponse, GuidanceResponse

import logging

logger = logging.getLogger("guidance-rpc")
logger.setLevel(logging.INFO)


def init_blocking(model_name: str | os.PathLike, hf_token: Optional[str] = None) -> guidance.llms.LLM:
    """
    Initialize a guidance model.

    :param hf_token: The huggingface token to use.
    :param model_name: The name of the huggingface model to use, or the path to a local model.
    :return: The initialized LLM.
    :raises:
        ValueError: If the model name is invalid or the model is already initialized.
    """

    logger.info("Initializing model %s", model_name)

    load_in_8bit: Optional[str] = os.environ.get("LOAD_IN_8BIT", None)
    repetition_penalty: Optional[str] = os.environ.get("REPETITION_PENALTY", None)
    top_p: Optional[str] = os.environ.get("TOP_P", None)
    top_k: Optional[str] = os.environ.get("TOP_K", None)
    temperature: Optional[str] = os.environ.get("TEMPERATURE", None)
    device_map: str = os.environ.get("DEVICE_MAP", "auto")

    hf_model: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        use_auth_token=hf_token,
        load_in_8bit=bool(load_in_8bit) if load_in_8bit is not None else None,
        repetition_penalty=float(repetition_penalty) if repetition_penalty is not None else None,
        top_p=float(top_p) if top_p is not None else None,
        top_k=float(top_k) if top_k is not None else None,
        temperature=float(temperature) if temperature is not None else None,
    )
    logger.info("Finished loading model %s", model_name)

    token_healing = bool(os.environ.get("TOKEN_HEALING", "True"))
    caching = bool(os.environ.get("CACHING", "True"))

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)

    return guidance.llms.Transformers(model=hf_model, tokenizer=hf_tokenizer, caching=caching, token_healing=token_healing)


async def init(model_name: str | os.PathLike, hf_token: Optional[str] = None) -> guidance.llms.LLM:
    """
    Initialize a guidance model.

    :param hf_token: The huggingface token to use.
    :param model_name: The name of the huggingface model to use, or the path to a local model.
    :return: The initialized LLM.
    :raises:
        ValueError: If the model name is invalid or the model is already initialized.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, init_blocking, model_name, hf_token)


class GuidanceServicer(guidance_pb2_grpc.GuidanceServicer):
    """
    The servicer for the guidance service.

    :param llm: The LLM to use.
    """

    def __init__(self, llm: Optional[guidance.llms.LLM]):
        self.llm = llm

    async def Guide(self, request, context) -> AsyncGenerator[GuidanceResponse, None]:

        if not self.llm:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("Model not initialized")
            raise ValueError("Model not initialized")

        logger.info("Program: %s", request.program)
        async for x in run(self.llm, request.program):
            yield GuidanceResponse(text=x)


class AsyncLoggingInterceptor(ServerInterceptor):
    async def intercept_service(
            self,
            continuation: Callable[
                [grpc.HandlerCallDetails], Awaitable[grpc.RpcMethodHandler]
            ],
            handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        logger.info("Received request: %s", handler_call_details.method)
        return await continuation(handler_call_details)


async def main() -> NoReturn:
    """
    The main entry point for the server.

    :returns (never): The function never returns.
    """
    logger.info("Starting server")
    guidance_servicer = GuidanceServicer(llm=None)
    server = grpc.aio.server(
        ThreadPoolExecutor(max_workers=10),
        interceptors=[AsyncLoggingInterceptor()]
    )

    health_servicer = grpc_health.v1.health.aio.HealthServicer()

    guidance__full_name = guidance_pb2_grpc.guidance__pb2.DESCRIPTOR.services_by_name["Guidance"].full_name

    await health_servicer.set(guidance__full_name, HealthCheckResponse.NOT_SERVING)
    await health_servicer.set("", HealthCheckResponse.NOT_SERVING)

    services: tuple[str, str] = (
        grpc_health.v1.health.SERVICE_NAME,
        guidance__full_name,
    )

    guidance_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    guidance_pb2_grpc.add_GuidanceServicer_to_server(guidance_servicer, server)
    reflection.enable_server_reflection(services, server)

    port = server.add_insecure_port('[::]:50051')
    logger.info("Starting server on port %d", port)
    await server.start()
    logger.info("Server started on port %d", port)

    model_name = os.environ.get("MODEL_NAME")

    if model_name is None:
        raise ValueError("MODEL_NAME is not set")

    llm = await init(model_name, os.environ.get("HF_TOKEN"))

    await health_servicer.set(guidance__full_name, HealthCheckResponse.SERVING)
    await health_servicer.set("", HealthCheckResponse.SERVING)

    guidance_servicer.llm = llm

    logger.info("Server initialized")
    await server.wait_for_termination()


async def run(llm: guidance.llms.LLM, template: str) -> AsyncGeneratorType[str, None]:
    """
    Run the given prompt on the given LLM and yield the results as they come in.

    :param llm: The LLM to run the prompt on.
    :param template: The prompt to run.

    :return: An async generator that yields the new strings as they come in.
    """
    program = guidance.Program(text=template, llm=llm, stream=True, async_mode=True)

    last: Optional[str] = None
    async for x in program():
        text: str = x.text
        if last is None:
            last = text
            yield text
        else:
            removed_prefix = text.removeprefix(last)
            yield removed_prefix
            print("sending: \"" + removed_prefix + "\"")
            last = text


if __name__ == '__main__':
    asyncio.run(main())
