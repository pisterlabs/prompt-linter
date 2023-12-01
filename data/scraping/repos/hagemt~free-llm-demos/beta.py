#!/usr/bin/env python3
import logging, sys  # noqa: E401
from dotenv import load_dotenv, find_dotenv
from langchain_experimental.agents.agent_toolkits.python.base import create_python_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.llms.openai import OpenAI

def main(*args, **kwargs) -> None:
    if args:
        print("--- Ask:", *args)
        llm = OpenAI(max_tokens=1000, temperature=0.25, **kwargs)
        cli = create_python_agent(llm=llm, tool=PythonREPLTool(), verbose=True)
        cli.run(*args)
    else:
        main("Find the roots (zeros) of the quadratic function: 3 * x**2 + 2*x - 1")
        print("NOTE(from another human programmer); expected answer is: (-1, +1/3)")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    load_dotenv(find_dotenv())
    main(*sys.argv[1:])
