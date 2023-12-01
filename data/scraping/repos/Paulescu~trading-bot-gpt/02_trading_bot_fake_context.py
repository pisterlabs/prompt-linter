import os
from typing import Dict, List

from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain
from dotenv import load_dotenv
from comet_llm import Span, end_chain, start_chain

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

def retrieve_technical_indicators() -> Dict[str, float]:
    """Retrieve technical indicators from a database."""
    return {'RSI': 0.5, 'MACD': 0.5, 'SMA': 0.5}


def retrieve_recent_news() -> List[str]:
    """Retrieve recent news from a database."""
    return [
        'ETH is about to upgrade its protocol',
        'DeFI sector is growing faster than expected.'
    ]

def get_llm_chain() -> LLMChain:

    # prompte template
    template = """
    I will give you the current technical indicators for ETH/USD and some recent financial news and you wil tell me if the asset price will go up or down in the next 1 hour.

    Dont tell me that predicting short-term price movements is highly uncertain please, but be brave!

    Format the output as a python dictionary with two keys: signal, and explanation. Signal is either -1 (meaning price goes down) or 1 (meaning price goes up). Explanation is a string exposing the reasoning behind your prediction. Please provide no more text in your response apart from this python dictionary.

    # technical indicators
    {technical_indicators}

    # news
    {recent_news}
    """
    prompt = PromptTemplate(template=template,
                            input_variables=["technical_indicators", "recent_news"])

    # llm
    llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

    # llm chain
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    return llm_chain

def main(current_time: int):

    start_chain(inputs={"current_time": current_time})

    # retrieve current technical indicators
    with Span(
        category="context-retrieval",
        name="Retrieve technical indicators",
        inputs={"current_time": current_time},
    ) as span:
        technical_indicators = retrieve_technical_indicators()
        span.set_outputs(outputs={"technical_indicators": technical_indicators})

    # retrieve recent news
    with Span(
        category="context-retrieval",
        name="Retrieve recent news",
        inputs={"current_time": current_time},
    ) as span:
        recent_news = retrieve_recent_news()
        span.set_outputs(outputs={"recent_news": recent_news})
    
    # llm that will reason about the context (technical indicators + recent news)
    # and output a prediction
    llm_chain = get_llm_chain()

    with Span(
        category="llm-reasoning",
        inputs={"technical_indicators": technical_indicators,"recent_news": recent_news},
    ) as span:
        response = llm_chain.run(technical_indicators=technical_indicators, recent_news=recent_news)
        print(response)
        span.set_outputs(outputs={"response": response})

    end_chain(outputs={"response": response})

from datetime import datetime

current_time = int(datetime.utcnow().timestamp())
main(current_time)