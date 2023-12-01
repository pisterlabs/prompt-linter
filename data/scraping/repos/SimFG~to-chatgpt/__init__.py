import json
import os
from typing import Optional

import cohere
from pydantic import BaseModel
from starlette.requests import Request

from to_chatgpt.common import BaseAdapter, convert_messages_to_prompt_without_role, logger, to_chatgpt_response, \
    to_chatgpt_response_stream

model_map = {
    "gpt-3.5-turbo": "command",
    "gpt-3.5-turbo-0301": "command-light",
    "gpt-4": "command-nightly",
    "gpt-4-0314": "command-light-nightly",
}


class CohereAdapter(BaseModel, BaseAdapter):
    client: Optional[cohere.Client] = None

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def openai_to_cohere_params(openai_params):
        model = model_map.get(openai_params["model"], "command")
        messages = openai_params["messages"]

        prompt = convert_messages_to_prompt_without_role(messages)

        cohere_params = {
            "model": model,
            "query": prompt,
        }

        if openai_params.get("max_tokens"):
            cohere_params["max_tokens"] = openai_params["max_tokens"]

        if openai_params.get("stop"):
            cohere_params["stop_sequences"] = openai_params.get("stop")

        if openai_params.get("temperature"):
            cohere_params["temperature"] = openai_params.get("temperature")

        if openai_params.get("top_p"):
            cohere_params["p"] = openai_params.get("top_p")

        if openai_params.get("stream"):
            cohere_params["stream"] = True

        return cohere_params

    async def achat(self, request: Request):
        openai_params = await request.json()
        cohere_params = self.openai_to_cohere_params(openai_params)
        logger.info(f"cohere_params:{cohere_params}")

        headers = request.headers
        auth_header = headers.get("authorization", None)
        if auth_header:
            token = auth_header.split(" ")[1]
            client = cohere.AsyncClient(token)
        else:
            if not self.client:
                self.client = cohere.AsyncClient(os.getenv("COHERE_API_KEY", ""))
            client = self.client

        is_stream = cohere_params.get("stream")
        try:
            if is_stream:
                response = await client.chat(**cohere_params)
                all_response_text = ""
                async for data in response:
                    all_response_text += data.text
                    yield to_chatgpt_response_stream(data.text, None)
                logger.info(f"cohere_stream_response:{json.dumps(all_response_text)}")
                yield to_chatgpt_response_stream("", "stop")
                yield "[DONE]"
            else:
                response = await client.chat(**cohere_params)
                logger.info(f"cohere_response:{response}")
                yield to_chatgpt_response(response.text)
        except Exception as e:
            logger.exception("cohere fail")
            if is_stream:
                yield to_chatgpt_response_stream(f"exception: {e}", None)
                yield to_chatgpt_response_stream("", "stop")
                yield "[DONE]"
            else:
                yield to_chatgpt_response(f"exception: {e}")
        finally:
            if auth_header:
                await client.close()
