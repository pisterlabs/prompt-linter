#!/usr/bin/env python
import logging

from dotenv import load_dotenv

from langchain.agents import AgentType, initialize_agent
from langchain.llms import OpenAI
from langchain.tools.yahoo_finance_news import YahooFinanceNewsTool

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

load_dotenv()


def main():
    logging.debug("Loading model ...")
    llm = OpenAI(temperature=0.2)
    # Prepare agent
    tools = [YahooFinanceNewsTool()]
    agent_chain = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    # Chain
    logging.debug("Start chain ...")
    response = agent_chain.run(
        "What happens today with EPAM stocks?",
    )

    logging.debug(f"Response: {response}")


if __name__ == "__main__":
    main()
