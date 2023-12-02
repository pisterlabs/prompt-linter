from time import sleep
from typing import Optional
from openai import OpenAI
from httpx import Client
from sympy import Symbol, Expr, symbols
import sym_pb2_grpc
from sym_pb2_grpc import SymServicer
from sym_pb2 import (
    ConvertMdRequest,
    ConvertMdReply,
    HelloReply,
    HelloRequest,
    ValueType,
)


class SymServer(SymServicer):
    def SayHello(self, request: HelloRequest, context):
        return HelloReply(message=f"Hello {request.name}!")

    def ConvertMdFormula(self, request: ConvertMdRequest, context):
        code = convert(request.md)
        return ConvertMdReply(sym=code)


SYSTEM = "You are a helpful assistant designed to convert formulas in markdown or latex \
to a python function signatured `formula() -> sympy.Expr` with python's sympy library. \
Remember, no args in the function signature. Remember, use `symbols` to create symbols in function \
first if needed."

from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    # In docker, do not need to set proxy, for it uses host network which does.
    # http_client=Client(proxies="http://127.0.0.1:7890"), timeout=30, max_retries=0
)


def convert(formula: str) -> str:
    q = f"You should only convert the correct side of ${formula}$ to a python function \
signatured `formula() -> sympy.Expr` with python's sympy library."
    print(f"Asking GPT: \n{q}")
    while True:
        try:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM,
                    },
                    {
                        "role": "user",
                        "content": q,
                    },
                ],
            )
            content = completion.choices[0].message.content
            print(f"GPT answer:\n {content}\n")
            code = extract(content if content else "")
            try:
                _ = run_code(code)
            except Exception as e:
                print(e)
                continue
        except Exception as e:
            print(e)
            sleep(20)
            continue
        break
    return code


def extract(content: str) -> str:
    flag = False
    ret = []
    for line in content.split("\n"):
        if line.startswith("```"):
            flag = not flag
            continue
        if flag:
            ret.append(line)
    return "\n".join(ret)


def run_code(code: str) -> Expr:
    buffer = {}
    print(f"exec: \n{code}\n")
    exec(code, buffer)
    expr = buffer["formula"]().factor()
    return expr


if __name__ == "__main__":
    print("Start test ...")
    code = convert(r"\sum_{i=1}^{n}x_{ij} = a_j")
    print(code)
    expr = run_code(code)
    print(expr)
