from typing import List
import logging
import asyncio

from langchain.chat_models import ChatAnthropic
from langchain.schema import StrOutputParser

from prompt import recommendation_prompt_template, questions, default_goal

logging.basicConfig(level=logging.INFO)


async def async_generate(runnable, html, question, goal):
    logging.info(question)
    resp = await runnable.ainvoke({"webpage_content": html, "task": question, "goal": goal})
    return resp


async def get_recommendations(html: str, goal: str) -> List[str]:
    if not goal:
        logging.info("Using default goal")
        goal = default_goal

    logging.info("Calling Claude.")
    model = ChatAnthropic(max_tokens_to_sample=1024, temperature=0)
    runnable = recommendation_prompt_template | model | StrOutputParser()
    calls = [async_generate(runnable, html, question, goal) for question in questions]
    recommendations = await asyncio.gather(*calls)

    logging.info(f"Response: {recommendations}")

    return recommendations
