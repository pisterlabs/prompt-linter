import json
# Import necessary modules for text processing, web scraping, and searching
from processing.text import summarize_text
from actions.web_scrape import scrape_text_with_selenium
from actions.web_search import web_search

# Import Langchain and related utilities for AI-driven chat and prompt management
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableMap, RunnableLambda
from langchain.schema.messages import SystemMessage
from agent.prompts import auto_agent_instructions, generate_search_queries_prompt
from config import Config

# Load global configurations
CFG = Config()

# [Initiation] Prepare the AI-driven message template for generating search queries
search_message = (generate_search_queries_prompt("{question}"))
SEARCH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "{agent_prompt}"),
    ("user", search_message)
])

# Load instructions for automated agent behavior
AUTO_AGENT_INSTRUCTIONS = auto_agent_instructions()
CHOOSE_AGENT_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content=AUTO_AGENT_INSTRUCTIONS),
    ("user", "task: {task}")
])

# [Content Retrieval and Summarization and Analysis] Define the process for scraping and summarizing text from a URL
scrape_and_summarize = {
    "question": lambda x: x["question"],
    "text": lambda x: scrape_text_with_selenium(x['url'])[1],
    "url": lambda x: x['url']
} | RunnableMap({
        "summary": lambda x: summarize_text(text=x["text"], question=x["question"], url=x["url"]),
        "url": lambda x: x['url']
}) | (lambda x: f"Source Url: {x['url']}\nSummary: {x['summary']}")

# Initialize a set to keep track of URLs that have already been seen to avoid duplicate content
seen_urls = set()

# [Web Search and Content Retrieval] Define the process for conducting multiple searches, avoiding duplicate URLs, and processing the results
multi_search = (
    lambda x: [
        {"url": url.get("href"), "question": x["question"]}
        for url in json.loads(web_search(query=x["question"], num_results=3))
        if not (url.get("href") in seen_urls or seen_urls.add(url.get("href")))
   ]
) | scrape_and_summarize.map() | (lambda x: "\n".join(x))

# Set up the search query and agent choice mechanisms using AI models
search_query = SEARCH_PROMPT | ChatOpenAI(model=CFG.smart_llm_model) | StrOutputParser() | json.loads
choose_agent = CHOOSE_AGENT_PROMPT | ChatOpenAI(model=CFG.smart_llm_model) | StrOutputParser() | json.loads

# [Initiation] Define how to get search queries based on agent prompts
get_search_queries = {
    "question": lambda x: x,
    "agent_prompt": {"task": lambda x: x} | choose_agent | (lambda x: x["agent_role_prompt"])
} | search_query


class GPTResearcherActor:
    # [Compilation and Output] Define the complete runnable process for the GPT Researcher, compiling all steps
    @property
    def runnable(self):
        return (
            get_search_queries
            | (lambda x: [{"question": q} for q in x])
            | multi_search.map()
            | (lambda x: "\n\n".join(x))
        )
