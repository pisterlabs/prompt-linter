import os
import typer
from rich import print, style
from rich.table import Table

import openai
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma

import credentials

os.environ["OPENAI_API_KEY"] = credentials.APIKEY

# True> save to disk the conversations and reuse the model. Good for repeated queries on the same data.
# False> always new queries.
PERSIST = False

def main():
    if PERSIST and os.path.exists("persist"):
        print("Reusing index...\n")
        vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
        index = VectorStoreIndexWrapper(vectorstore=vectorstore)
    else:
        loader = DirectoryLoader("mydata/")
        if PERSIST:
            index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
        else:
            index = VectorstoreIndexCreator().from_loaders([loader])
    
    chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        retriever=index.vectorstore.as_retriever(search_kwargs={"k": 2}),
    )

    print("[bold green]Welcome. Ask any questions about the content of your documents stored in the mydata directory. And here are a couple of useful commands.[/bold green]")
    
    table = Table("Comand", "Description")
    table.add_row("exit", "Close the app")
    table.add_row("new", "New chat")
    
    print(table)
    
    while True:
        query = prompt()
        if query == "new":
            print("[bold green]New chat created[/bold green]")
            query = prompt()
        print(chain.run(query))

def prompt() -> str:
    prompt = typer.prompt("\n>> ")
    if prompt == "exit":
        confirmation = typer.confirm("[bold red]¿Are you sure?[/bold red]")
        if confirmation:
            print("Hasta la vista Baby!")
            raise typer.Abort()
    return prompt

if __name__ == "__main__":
    typer.run(main)

