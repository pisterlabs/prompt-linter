import os
from pprint import pprint

from langchain.utilities import GoogleSearchAPIWrapper


def run_google_sce_example():
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    os.environ["GOOGLE_CSE_ID"] = os.getenv("GOOGLE_SCE_ID")
    search = GoogleSearchAPIWrapper()
    # result = search.run("What is LangChain?")
    results = search.results("What is LangChain?", num_results=10)
    pprint(results)
