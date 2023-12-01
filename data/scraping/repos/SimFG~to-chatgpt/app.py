import argparse
from typing import Optional

import uvicorn
from fastapi import Request

from to_chatgpt import common
from to_chatgpt.common import BaseAdapter, init_app

adapter: Optional[BaseAdapter] = None
app = init_app()
server_name = "new_bing"


@app.get("/")
async def hello():
    return f"hello, {server_name} to chatgpt server"


@app.api_route(
    "/v1/chat/completions", methods=["POST", "OPTIONS"],
)
async def chat(request: Request):
    return await common.achat(adapter, request)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--host", default="0.0.0.0", help="the hostname to listen on"
    )
    parser.add_argument(
        "-p", "--port", type=int, default=8000, help="the port to listen on"
    )
    parser.add_argument(
        "-a", "--adapter", default=server_name, help="the name of server adapter"
    )
    parser.add_argument("-l", "--log", default="debug", help="the log level")

    args = parser.parse_args()

    global adapter
    if args.adapter == "new_bing":
        from to_chatgpt.new_bing import NewBingAdapter

        adapter = NewBingAdapter()
    elif args.adapter == "claude":
        from to_chatgpt.claude import ClaudeAdapter

        adapter = ClaudeAdapter()
    elif args.adapter == "cohere":
        from to_chatgpt.cohere import CohereAdapter

        adapter = CohereAdapter()
    else:
        raise ValueError(f"unknown adapter: {args.adapter}")

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log)


if __name__ == "__main__":
    run()
