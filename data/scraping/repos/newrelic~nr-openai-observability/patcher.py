import inspect
import logging
import sys
import time
import uuid
from argparse import ArgumentError

import openai
from newrelic_telemetry_sdk import Span

import nr_openai_observability.consts as consts
from nr_openai_observability.build_events import (
    build_completion_error_events,
    build_completion_events,
    build_embedding_error_event,
    build_embedding_event,
)
from nr_openai_observability.error_handling_decorator import handle_errors
from nr_openai_observability.openai_monitoring import monitor
from nr_openai_observability.stream_patcher import (
    patcher_create_chat_completion_stream,
    patcher_create_chat_completion_stream_async,
)

logger = logging.getLogger("nr_openai_observability")


def _patched_call(original_fn, patched_fn, stream_patched_fn=None):
    if hasattr(original_fn, "is_patched_by_monitor"):
        return original_fn

    def _inner_patch(*args, **kwargs):
        try:
            if kwargs.get("stream") is True and stream_patched_fn is not None:
                return stream_patched_fn(original_fn, *args, **kwargs)
            else:
                return patched_fn(original_fn, *args, **kwargs)
        except Exception as ex:
            raise ex

    _inner_patch.is_patched_by_monitor = True

    return _inner_patch


def _patched_call_async(original_fn, patched_fn, stream_patched_fn=None):
    if hasattr(original_fn, "is_patched_by_monitor"):
        return original_fn

    async def _inner_patch(*args, **kwargs):
        try:
            if kwargs.get("stream") is True and stream_patched_fn is not None:
                return await stream_patched_fn(original_fn, *args, **kwargs)
            else:
                return await patched_fn(original_fn, *args, **kwargs)
        except Exception as ex:
            raise ex

    _inner_patch.is_patched_by_monitor = True

    return _inner_patch


def patcher_convert_to_openai_object(original_fn, *args, **kwargs):
    response = original_fn(*args, **kwargs)

    if isinstance(args[0], openai.openai_response.OpenAIResponse):
        setattr(response, "_nr_response_headers", getattr(args[0], "_headers", {}))

    return response


def patcher_create_chat_completion(original_fn, *args, **kwargs):
    logger.debug(
        f"Running the original function: '{original_fn.__qualname__}'. args:{args}; kwargs: {kwargs}"
    )

    result, time_delta = None, None
    span = monitor.create_span()
    try:
        timestamp = time.time()
        result = original_fn(*args, **kwargs)
        time_delta = time.time() - timestamp
    except Exception as ex:
        span.finish()
        handle_create_chat_completion(result, kwargs, ex, time_delta, span)
        raise ex
    span.finish()

    logger.debug(f"Finished running function: '{original_fn.__qualname__}'.")

    return handle_create_chat_completion(result, kwargs, None, time_delta, span)


async def patcher_create_chat_completion_async(original_fn, *args, **kwargs):
    logger.debug(
        f"Running the original function: '{original_fn.__qualname__}'. args:{args}; kwargs: {kwargs}"
    )
    result, time_delta = None, None
    span = monitor.create_span()
    try:
        timestamp = time.time()
        result = await original_fn(*args, **kwargs)
        time_delta = time.time() - timestamp
    except Exception as ex:
        span.finish()
        handle_create_chat_completion(result, kwargs, ex, time_delta, span)
        raise ex
    span.finish()

    logger.debug(f"Finished running function: '{original_fn.__qualname__}'.")

    return handle_create_chat_completion(result, kwargs, None, time_delta, span)


@handle_errors
def handle_create_chat_completion(
    response, request, error, response_time, span: Span = None
):
    events = None
    if error:
        events = build_completion_error_events(request, error)
    else:
        events = build_completion_events(
            response, request, getattr(response, "_nr_response_headers"), response_time
        )
        delattr(response, "_nr_response_headers")

    for event in events["messages"]:
        monitor.record_event(event, consts.MessageEventName)
    monitor.record_event(events["completion"], consts.SummaryEventName)
    if span:
        span["attributes"].update(events["completion"])
        span["attributes"]["name"] = consts.SummaryEventName
        monitor.record_span(span)

    return response


def get_arg_value(
    args,
    kwargs,
    pos,
    kw,
):
    try:
        return kwargs[kw]
    except KeyError:
        try:
            return args[pos]
        except IndexError:
            raise ArgumentError("Missing required argument: %s" % (kw,))


@handle_errors
def handle_similarity_search(
    original_fn, response, request_args, request_kwargs, error, response_time
):
    vendor = inspect.getmodule(original_fn).__name__.split(".")[-1]
    query = get_arg_value(request_args, request_kwargs, 1, "query")
    k = request_kwargs.get("k")
    if k is None and len(request_args) >= 3:
        k = request_args[1]

    event_dict = {
        "provider": vendor,
        "query": query,
        "k": k,
        "response_time": response_time,
    }

    if error:
        event_dict["error"] = str(error)
    else:
        documents = response
        event_dict["search_id"] = str(uuid.uuid4())
        event_dict["document_count"] = len(documents)
        for idx, document in enumerate(documents):
            result_event_dict = {
                "search_id": event_dict["search_id"],
                "result_rank": idx,
                "document_page_content": str(document.page_content),
            }
            for kwarg_key, v in document.metadata.items():
                result_event_dict[f"document_metadata_{kwarg_key}"] = str(v)

            result_event_dict.update(**event_dict)

            monitor.record_event(result_event_dict, consts.VectorSearchResultsEventName)

    monitor.record_event(event_dict, consts.VectorSearchEventName)

    return response


async def patcher_create_completion_async(original_fn, *args, **kwargs):
    logger.debug(
        f"Running the original function: '{original_fn.__qualname__}'. args:{args}; kwargs: {kwargs}"
    )

    timestamp = time.time()
    result = await original_fn(*args, **kwargs)
    time_delta = time.time() - timestamp

    logger.debug(f"Finished running function: '{original_fn.__qualname__}'.")

    return handle_create_completion(result, time_delta, **kwargs)


def patcher_create_completion(original_fn, *args, **kwargs):
    logger.debug(
        f"Running the original function: '{original_fn.__qualname__}'. args:{args}; kwargs: {kwargs}"
    )

    timestamp = time.time()
    logger.debug(
        f"Running the original function: '{original_fn.__qualname__}'. args:{args}; kwargs: {kwargs}"
    )

    timestamp = time.time()
    result = original_fn(*args, **kwargs)
    time_delta = time.time() - timestamp

    logger.debug(f"Finished running function: '{original_fn.__qualname__}'.")

    return handle_create_completion(result, time_delta, **kwargs)


@handle_errors
def handle_create_completion(response, time_delta, **kwargs):
    def flatten_dict(dd, separator=".", prefix="", index=""):
        if len(index):
            index = index + separator
        return (
            {
                prefix + separator + index + k if prefix else k: v
                for kk, vv in dd.items()
                for k, v in flatten_dict(vv, separator, kk).items()
            }
            if isinstance(dd, dict)
            else {prefix: dd}
        )

    choices_payload = {}
    for i, choice in enumerate(response.get("choices")):
        choices_payload.update(flatten_dict(choice, prefix="choices", index=str(i)))

    logger.debug(dict(**kwargs))

    event_dict = {
        **kwargs,
        "response_time": time_delta,
        **flatten_dict(response.to_dict_recursive(), separator="."),
        **choices_payload,
    }
    event_dict.pop("choices")

    if "messages" in event_dict:
        event_dict["messages"] = str(kwargs.get("messages"))

    logger.debug(f"Reported event dictionary:\n{event_dict}")
    monitor.record_event(event_dict)

    return response


def patcher_create_embedding(original_fn, *args, **kwargs):
    logger.debug(
        f"Running the original function: '{original_fn.__qualname__}'. args:{args}; kwargs: {kwargs}"
    )

    result, time_delta = None, None
    try:
        timestamp = time.time()
        result = original_fn(*args, **kwargs)
        time_delta = time.time() - timestamp
    except Exception as ex:
        handle_create_embedding(result, kwargs, ex, time_delta)
        raise ex

    logger.debug(f"Finished running function: '{original_fn.__qualname__}'.")

    return handle_create_embedding(result, kwargs, None, time_delta)


async def patcher_create_embedding_async(original_fn, *args, **kwargs):
    logger.debug(
        f"Running the original function: '{original_fn.__qualname__}'. args:{args}; kwargs: {kwargs}"
    )

    result, time_delta = None, None
    try:
        timestamp = time.time()
        result = await original_fn(*args, **kwargs)
        time_delta = time.time() - timestamp
    except Exception as ex:
        handle_create_embedding(result, kwargs, ex, time_delta)
        raise ex

    logger.debug(f"Finished running function: '{original_fn.__qualname__}'.")

    return handle_create_embedding(result, kwargs, None, time_delta)


def patcher_similarity_search(original_fn, *args, **kwargs):
    logger.debug(
        f"Running the original function: '{original_fn.__qualname__}'. args:{args}; kwargs: {kwargs}"
    )

    result, time_delta = None, None
    try:
        timestamp = time.time()
        result = original_fn(*args, **kwargs)
        time_delta = time.time() - timestamp
    except Exception as ex:
        handle_similarity_search(original_fn, result, args, kwargs, ex, time_delta)
        raise ex

    logger.debug(f"Finished running function: '{original_fn.__qualname__}'.")

    return handle_similarity_search(original_fn, result, args, kwargs, None, time_delta)


@handle_errors
def handle_create_embedding(response, request, error, response_time):
    event = None
    if error:
        event = build_embedding_error_event(request, error)
    else:
        event = build_embedding_event(
            response, request, getattr(response, "_nr_response_headers"), response_time
        )
        delattr(response, "_nr_response_headers")

    monitor.record_event(event, consts.EmbeddingEventName)

    return response


def perform_patch_langchain_vectorstores():
    import langchain.vectorstores
    from langchain.vectorstores import __all__ as langchain_vectordb_list

    for vector_store in langchain_vectordb_list:
        try:
            cls = getattr(langchain.vectorstores, vector_store)
            cls.similarity_search = _patched_call(
                cls.similarity_search, patcher_similarity_search
            )
        except AttributeError:
            pass


def perform_patch():
    try:
        openai.Embedding.create = _patched_call(
            openai.Embedding.create, patcher_create_embedding
        )
    except AttributeError:
        pass

    try:
        openai.Embedding.acreate = _patched_call_async(
            openai.Embedding.acreate, patcher_create_embedding_async
        )
    except AttributeError:
        pass

    try:
        openai.Completion.create = _patched_call(
            openai.Completion.create, patcher_create_completion
        )
    except AttributeError:
        pass

    try:
        openai.Completion.acreate = _patched_call_async(
            openai.Completion.acreate, patcher_create_completion_async
        )
    except AttributeError:
        pass

    try:
        openai.ChatCompletion.create = _patched_call(
            openai.ChatCompletion.create,
            patcher_create_chat_completion,
            patcher_create_chat_completion_stream,
        )
    except AttributeError:
        pass

    try:
        openai.ChatCompletion.acreate = _patched_call_async(
            openai.ChatCompletion.acreate,
            patcher_create_chat_completion_async,
            patcher_create_chat_completion_stream_async,
        )
    except AttributeError:
        pass

    try:
        openai.util.convert_to_openai_object = _patched_call(
            openai.util.convert_to_openai_object, patcher_convert_to_openai_object
        )
    except AttributeError:
        pass

    if "langchain" in sys.modules:
        perform_patch_langchain_vectorstores()
