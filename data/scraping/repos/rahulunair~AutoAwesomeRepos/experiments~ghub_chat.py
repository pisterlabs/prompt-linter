import csv
import sys
from typing import Optional

import typer
from langchain.agents import create_csv_agent
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from rich import print

csv.field_size_limit(sys.maxsize)


csv_file = "./results/filtered_readme_repos_20230329-112511.csv"
loader = CSVLoader(file_path=csv_file)
print(type(loader))
data = loader.load()


app = typer.Typer()


def index_query(query: str) -> str:
    index = VectorstoreIndexCreator().from_loaders([loader])
    return index.query(query)


def agent_run(
    query: str,
    previous_query: Optional[str] = None,
    previous_answer: Optional[str] = None,
) -> str:
    agent = create_csv_agent(OpenAI(temperature=0), csv_file, verbose=True)
    return agent.run(query)


@app.command()
def qa(query: str):
    response = index_query(query)
    print(response)


@app.command()
def agent():
    previous_query = None
    previous_answer = None

    while True:
        query = typer.prompt("Enter your question (type 'exit' to quit):")
        if query.lower() == "exit":
            break

        if previous_query and previous_answer:
            response = agent_run(query, previous_query, previous_answer)
        else:
            response = agent_run(query)

        print(response)
        previous_query = query
        previous_answer = response


if __name__ == "__main__":
    app()
