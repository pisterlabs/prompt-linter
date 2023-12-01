#!/usr/bin/env python
import logging
from langchain.llms import GPT4All
from langchain.prompts import PromptTemplate
import workaround
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main():
    logging.debug("Loading model ...")
    with workaround.suppress_stdout_stderr():
        llm = GPT4All(
            model="./models/llama-2-7b-chat.ggmlv3.q4_0.bin",
        )
    # Prepare agent
    tools = load_tools(["wikipedia"], llm=llm)
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )

    # Prepare query
    prompt_template = PromptTemplate.from_template(
        "Tell me about {content}. Do not exceed 42 tokens."
    )
    question = prompt_template.format(content="Hello World!")

    # Chain
    logging.debug("Start chain ...")
    response = agent.run(question)

    logging.debug(f"Response: {response}")


if __name__ == "__main__":
    main()
