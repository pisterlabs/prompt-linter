import json
import sys
import time
import requests
from openai.openai_response import OpenAIResponse

def log_call(self, args, kwargs, response, cached):
    GREEN = "\033[32m"
    RED = "\033[31m"
    END = "\033[0m"
    BLUE = "\033[34m"
    prompt = "\n".join(
        f'{message["name"] if "name" in message else message["role"]}: {RED if message["role"] == "user" else BLUE}{message["content"]}{END}'
        for message in kwargs["params"]["messages"]
    )
    params = {k: v for k, v in kwargs["params"].items() if k != "messages"}
    message = response.data["choices"][0]["message"]
    if "function_call" in message:
        response = message["function_call"]
    else:
        response = message["content"]
    sys.stderr.write(
        f"{'>'*8}\n{prompt}\n{params}\n{'='*8}\n{GREEN}{response}{END}\n{'<'*8}\n\n"
    )


def request_with_cache(self, *args, **kwargs):
    """Cache the result of a function call."""
    from hashlib import md5
    from pathlib import Path

    cache_dir = Path("openai_cache")
    cache_duration = 3600

    data = {
        "args": list(args),
        "kwargs": kwargs,
    }
    name = md5(json.dumps(data).encode()).hexdigest()
    filename = cache_dir / f"{name}.json"
    if filename.exists():
        with filename.open() as f:
            cache = json.load(f)
            if time.time() < cache["timestamp"] + cache_duration:
                headers = requests.structures.CaseInsensitiveDict(cache["_headers"])
                response = OpenAIResponse(cache["data"], headers)
                self.log_call(args, kwargs, response, True)
                return response, cache["got_stream"], cache["api_key"]

    response, got_stream, api_key = self.request_without_cache(*args, **kwargs)
    data["timestamp"] = time.time()
    data["_headers"] = dict(response._headers)
    data["data"] = response.data
    data["got_stream"] = got_stream
    data["api_key"] = api_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    with filename.open("w") as f:
        json.dump(data, f)
    self.log_call(args, kwargs, response, False)
    return response, got_stream, api_key
