import sys
import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, List
from urllib.parse import urlparse, parse_qs
from sentence_transformers import SentenceTransformer
from memory import Memory

from llama_attn_hijack import hijack_llama_attention_xformers

if Path("./guidance").exists():
    sys.path.insert(0, str(Path("guidance")))
    print("Adding ./guidance to import tree.")

import guidance

MODEL_EXECUTOR: str = None
model_name: str = None


# EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
# EMBEDDING_MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"
# EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_MODEL_NAME = "intfloat/e5-small-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

memory = None


def setup_models(model_name: str):
    """
    model_name: a Huggingface path like TheBlock/tulu-13B-GPTQ
    """

    # A slight improvement in memory usage by using xformers attention:
    if "--xformers" in sys.argv:
        hijack_llama_attention_xformers()

    model = None

    print(f"Loading model with executor: {MODEL_EXECUTOR}")

    if MODEL_EXECUTOR == "autogptq":
        from llama_autogptq import LLaMAAutoGPTQ

        model = LLaMAAutoGPTQ(model_name)
    elif MODEL_EXECUTOR == "gptq":
        from llama_gptq import LLaMAGPTQ

        model = LLaMAGPTQ(model_name)
    elif MODEL_EXECUTOR == "transformersgptq":
        from llama_transformers_autogptq import LLaMATransformersAutoGPTQ

        model = LLaMATransformersAutoGPTQ(model_name)
    elif MODEL_EXECUTOR == "exllama":
        from llama_exllama import ExLLaMA

        model = ExLLaMA(model_name)
    elif MODEL_EXECUTOR == "ctransformers":
        from llama_ctransformers import LLaMATransformer

        model = LLaMATransformer(model_name)
    elif MODEL_EXECUTOR == "transformers":
        from llama_transformer import LLaMATransformer

        model = LLaMATransformer(model_name)
    elif MODEL_EXECUTOR == "llamacpp":
        # from llama_cpp_hf import LlamacppHF
        # model = LlamacppHF(model_name)
        from llama_cpp_guidance.llm import LlamaCpp

        model = LlamaCpp(model_path=f'models/{model_name}', n_gpu_layers=9999, n_ctx=4096)
    elif MODEL_EXECUTOR == "awq":
        from llama_awk import LLaMAAwk

        model = LLaMAAwk(model_name)
    elif MODEL_EXECUTOR == "openai":
        model = guidance.llms.OpenAI(endpoint='http://localhost:8002', model='TheBloke/Llama-2-7b-Chat-AWQ')

    # global memory
    # memory = Memory(embedding_model)
    print("Memory initialized.")

    guidance.llms.Transformers.cache.clear()
    guidance.llm = model
    # print(f"Token healing enabled: {guidance.llm.token_healing}")


class MyHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        print(f"POST {self.path}")

        if self.path == "/embeddings":
            self.handle_embeddings()
        elif self.path == "/chat":
            self.handle_chat_streaming()
        elif self.path == "/memory":
            self.handle_memory_post()

    def do_GET(self):
        print(f"GET {self.path}")

        if self.path.startswith("/memory"):
            self.handle_memory_get()
        elif self.path.startswith("/model"):
            self.handle_model_get()

    def handle_model_get(self):
        print('model get requested')

        # Send response headers
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()

        result_json = json.dumps({ 'model_name': model_name }).encode("utf-8")

        self.wfile.write(result_json)

    def handle_memory_get(self):
        print("memory get requested")

        parsed_path = urlparse(self.path)
        path = parsed_path.path
        params = parse_qs(parsed_path.query)

        # For example, let's print the parsed values
        print("Path:", path)
        print("Parameters:", params)

        # Send response headers
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()

        query: str = params["query"]
        kind: str = params.get("kind", "")
        top_n: int = int(params.get("top_n", "3"))

        filters = dict()
        if kind != "":
            filters["kind"] = kind

        result = memory.query(query, top_n, filters)
        result_json = json.dumps(result).encode("utf-8")

        print(f"result: {result}")
        print(f"result json: {result_json}")

        self.wfile.write(result_json)

    def handle_memory_post(self):
        print("memory write requested")
        content_length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(content_length).decode("utf-8"))

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

        ids: List[str] = body["ids"]
        documents: List[str] = body["documents"]
        metadatas: List[Dict[str, str]] = body["metadatas"]

        if len(ids) != len(documents) != len(metadatas):
            print(
                f"Warning: body count mismatch: {len(ids)}, {len(documents)}, {len(metadatas)}"
            )
            self.send_error(400)

            return

        memory.add(ids[0], documents[0], metadatas[0])

    def handle_embeddings(self):
        print("embeddings requested")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

        content_length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(content_length).decode("utf-8"))

        text = body["input"] if "input" in body else body["text"]

        if type(text) is str:
            text = [text]

        embeddings = embedding_model.encode(text).tolist()

        data = [
            {"object": "embedding", "embedding": emb, "index": n}
            for n, emb in enumerate(embeddings)
        ]

        response = json.dumps(
            {
                "object": "list",
                "data": data,
                "model": EMBEDDING_MODEL_NAME,
                "usage": {
                    "prompt_tokens": 0,
                    "total_tokens": 0,
                },
            }
        )

        self.wfile.write(response.encode("utf-8"))

    def handle_chat_streaming(self):
        print("streaming chat requested")
        guidance.llms.Transformers.cache.clear()

        content_length = int(self.headers["Content-Length"])  # Get the size of data
        post_data = self.rfile.read(content_length).decode("utf-8")
        data = json.loads(post_data)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        # self.send_header('Connection', 'keep-alive')
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        template: str = data["template"]
        parameters: dict = data["parameters"]

        # g = guidance(template, stream=True, caching=False)
        g = guidance(template, stream=False)

        output_skips = {}

        skip_text = 0

        print("streaming model output...")
        # for output in g(**parameters):
        output = g(**parameters)
        print("\n...streaming chunk...")

        filtered_variables = dict()

        for key, val in output.variables().items():
            if (
                isinstance(val, str)
                and key not in parameters.keys()
                and key != "llm"
                and key != "@raw_prefix"
            ):
                filtered_variables[key] = val
                # skip = output_skips.get(key, 0)
                # filtered_variables[key] = val[skip:]
                # output_skips[key] = len(val)

        print(f"variables: {filtered_variables}")

        response = {
            "text": output.text[skip_text:],
            "variables": filtered_variables,
        }

        print(f"response:\n{response}")

        skip_text = len(output.text)

        response_str = f"data: {json.dumps(response).rstrip()}\n\n"
        sys.stdout.flush()
        self.wfile.write("event: streamtext\n".encode("utf-8"))
        self.wfile.write(response_str.encode("utf-8"))
        self.wfile.flush()

        print("done getting output from model.")


def run(server_class=HTTPServer, handler_class=MyHandler):
    global MODEL_EXECUTOR

    if len(sys.argv) < 3:
        raise Exception(
            "Expected to be invoked with two arguments: model_name and executor"
        )

    global model_name
    model_name = sys.argv[1]

    MODEL_EXECUTOR = sys.argv[2]

    setup_models(model_name)

    port = 8000
    server_address = ("0.0.0.0", port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting httpd. Listening on port {port}...\n")
    httpd.serve_forever()


if __name__ == "__main__":
    run()
