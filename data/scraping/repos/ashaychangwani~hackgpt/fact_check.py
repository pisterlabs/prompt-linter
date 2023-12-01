from langchain.chains import LLMSummarizationCheckerChain
from langchain.chat_models import PromptLayerChatOpenAI
import os
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import load_tools
from langchain.retrievers.document_compressors import CohereRerank

import io
import contextlib

class FactChecker:
    def __init__(self):
        x=3
        llm = PromptLayerChatOpenAI(temperature=0.7, model_name = 'gpt-3.5-turbo', openai_api_key=os.getenv("OPENAI_API_KEY"))
        tools = load_tools(["serpapi", "llm-math"], llm=llm, serpapi_api_key=os.getenv("SERPAPI_API_KEY"))
        self.agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
        self.prompt_factcheck = f"""
        You are a professional article fact checker. 
        Can you fact check this article: <text>

        Perform the fact check by listing down the "factual" statements that the article author claim to be true into bullet points, and present this points.
        Then for each point, find out whether they are true by cross checking with other websites.
        Finally, present the end result as a list in this format:
        - <Statement> : <Verdict> (Source)
        """
        self.compressor = CohereRerank()

    
    def check(self, text):
        try:
            article_final = self.agent.run(self.prompt_factcheck.replace("<text>", text))
            return article_final
        except:
            return self.check(text)