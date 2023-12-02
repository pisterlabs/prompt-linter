import openai
from litellm import completion
import os
from proxy_app.utils import (
    price_calculator_chat_completion,
    price_calculator_embedding_completion,
    MAX_TOKEN_LIMIT,
)
import dotenv
import time

dotenv.load_dotenv(".env")

openai.api_key = os.environ.get("OPENAI_API_KEY")


class ChatCompletionHandler:
    def __init__(self, req, url):
        self.url = url
        self.req = req
        self.prompt_tokens = 0
        self.response_tokens = 0
        self.total_tokens = 0
        self.tokens_cost = 0

    def makeRequest(self):
        try:
            if "stream" in self.req and self.req["stream"] == True:
                return True, "", True
            else:
                try:
                    self.req["max_tokens"] = min(
                        int(self.req["max_tokens"]), MAX_TOKEN_LIMIT
                    )
                except KeyError:
                    self.req["max_tokens"] = MAX_TOKEN_LIMIT
                # start timer
                timeStart = time.time()
                response = completion(**self.req)
                # end timer
                timeEnd = time.time()
                # calculate time taken
                timeTaken = timeEnd - timeStart
                print(f"Time taken (core API call): {timeTaken}")
                self.prompt_tokens = response["usage"]["prompt_tokens"]
                self.response_tokens = response["usage"]["completion_tokens"]
                self.total_tokens = response["usage"]["total_tokens"]
                self.tokens_cost = price_calculator_chat_completion(response["usage"])
                return True, response, False

        except KeyError:
            try:
                self.req["max_tokens"] = min(
                    int(self.req["max_tokens"]), MAX_TOKEN_LIMIT
                )
            except KeyError:
                self.req["max_tokens"] = MAX_TOKEN_LIMIT

            response = completion(**self.req)

            self.prompt_tokens = response["usage"]["prompt_tokens"]
            self.response_tokens = response["usage"]["completion_tokens"]
            self.total_tokens = response["usage"]["total_tokens"]
            self.tokens_cost = completion_cost(completion_response=response)
            return True, response, False

        except openai.InvalidRequestError as e:
            return False, str(e), False


class EmbeddingHandler:
    def __init__(self, req, url):
        self.url = url
        self.req = req
        self.prompt_tokens = 0
        self.total_tokens = 0
        self.tokens_cost = 0

    def makeRequest(self):
        try:
            self.req["max_tokens"] = min(int(self.req["max_tokens"]), MAX_TOKEN_LIMIT)
        except KeyError:
            self.req["max_tokens"] = MAX_TOKEN_LIMIT

        try:
            response = openai.Embedding.create(**self.req)
        except openai.InvalidRequestError as e:
            return False, str(e)

        self.prompt_tokens = response["usage"]["prompt_tokens"]
        self.total_tokens = response["usage"]["total_tokens"]
        self.tokens_cost = price_calculator_embedding_completion(response["usage"])
        return True, response
