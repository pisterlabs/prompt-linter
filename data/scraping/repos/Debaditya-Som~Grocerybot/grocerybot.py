# # Automatically restart kernel after installs so that your environment can access the new packages
# import IPython

# app = IPython.Application.instance()
# app.kernel.do_shutdown(True)
# !pip install --upgrade google-cloud-aiplatform==1.27.0 langchain==0.0.196 faiss-cpu==1.7.4 --user

import glob
import pprint
from typing import Any, Iterator, List

from langchain.agents import AgentType, initialize_agent
from langchain.document_loaders import TextLoader
from langchain.embeddings import VertexAIEmbeddings
from langchain.llms import VertexAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.tools import tool
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStoreRetriever
from tqdm import tqdm

llm = VertexAI(
    model_name="text-bison@001",
    max_output_tokens=256,
    temperature=0,
    top_p=0.8,
    top_k=40,
)

embedding = VertexAIEmbeddings()

#First, fetch the dummy data on products and recipes from a public Cloud Storage bucket and store it locally.

# !gsutil -m cp -r "gs://github-repo/use-cases/grocery_bot/*" .

def chunks(lst: List[Any], n: int) -> Iterator[List[Any]]:
    """Yield successive n-sized chunks from lst.

    Args:
        lst: The list to be chunked.
        n: The size of each chunk.

    Yields:
        A list of the next n elements from lst.
    """

    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def load_docs_from_directory(dir_path: str) -> List[Document]:
    """Loads a series of docs from a directory.

    Args:
      dir_path: The path to the directory containing the docs.

    Returns:
      A list of the docs in the directory.
    """

    docs = []
    for file_path in glob.glob(dir_path):
        loader = TextLoader(file_path)
        docs = docs + loader.load()
    return docs


def create_retriever(top_k_results: int, dir_path: str) -> VectorStoreRetriever:
    """Create a recipe retriever from a list of top results and a list of web pages.

    Args:
        top_k_results: number of results to return when retrieving
        dir_path: List of web pages.

    Returns:
        A recipe retriever.
    """

    BATCH_SIZE_EMBEDDINGS = 5
    docs = load_docs_from_directory(dir_path=dir_path)
    doc_chunk = chunks(docs, BATCH_SIZE_EMBEDDINGS)
    for index, chunk in tqdm(enumerate(doc_chunk)):
        if index == 0:
            db = FAISS.from_documents(chunk, embedding)
        else:
            db.add_documents(chunk)
    retriever = db.as_retriever(search_kwargs={"k": top_k_results})
    return retriever

recipe_retriever = create_retriever(top_k_results=2, dir_path="./recipes/*")
product_retriever = create_retriever(top_k_results=5, dir_path="./products/*")

docs = recipe_retriever.get_relevant_documents("Any lasagne recipes?")
pprint.pprint([doc.metadata for doc in docs])


@tool(return_direct=True)
def retrieve_recipes(query: str) -> str:
    """
    Searches the recipe catalog to find recipes for the query.
    Return the output without processing further.
    """
    docs = recipe_retriever.get_relevant_documents(query)

    return (
        f"Select the recipe you would like to explore further about {query}: [START CALLBACK FRONTEND] "
        + str([doc.metadata for doc in docs])
        + " [END CALLBACK FRONTEND]"
    )


@tool(return_direct=True)
def retrieve_products(query: str) -> str:
    """Searches the product catalog to find products for the query.
    Use it when the user asks for the products available for a specific item. For example `Can you show me which onions I can buy?`
    """
    docs = product_retriever.get_relevant_documents(query)
    return (
        f"I found these products about {query}:  [START CALLBACK FRONTEND] "
        + str([doc.metadata for doc in docs])
        + " [END CALLBACK FRONTEND]"
    )


@tool
def recipe_selector(path: str) -> str:
    """
    Use this when the user selects a recipe.
    You will need to respond to the user telling what are the options once a recipe is selected.
    You can explain what are the ingredients of the recipe, show you the cooking instructions or suggest you which products to buy from the catalog!
    """
    return "Great choice! I can explain what are the ingredients of the recipe, show you the cooking instructions or suggest you which products to buy from the catalog!"


docs = load_docs_from_directory("./recipes/*")
recipes_detail = {doc.metadata["source"]: doc.page_content for doc in docs}


@tool
def get_recipe_detail(path: str) -> str:
    """
    Use it to find more information for a specific recipe, such as the ingredients or the cooking steps.
    Use this to find what are the ingredients for a recipe or the cooking steps.

    Example output:
    Ingredients:

    * 1 pound lasagna noodles
    * 1 pound ground beef
    * 1/2 cup chopped onion
    * 2 cloves garlic, minced
    * 2 (28 ounce) cans crushed tomatoes
    * 1 (15 ounce) can tomato sauce
    * 1 teaspoon dried oregano

    Would you like me to show you the suggested products from the catalogue?
    """
    try:
        return recipes_detail[path]
    except KeyError:
        return "Could not find the details for this recipe"



@tool(return_direct=True)
def get_suggested_products_for_recipe(recipe_path: str) -> str:
    """Use this only if the user would like to buy certain products connected to a specific recipe example 'Can you give me the products I can buy for the lasagne?'",

    Args:
        recipe_path: The recipe path.

    Returns:
        A list of products the user might want to buy.
    """
    recipe_to_product_mapping = {
        "./recipes/lasagne.txt": [
            "./products/angus_beef_lean_mince.txt",
            "./products/large_onions.txt",
            "./products/classic_carrots.txt",
            "./products/classic_tomatoes.txt",
        ]
    }

    return (
        "These are some suggested ingredients for your recipe [START CALLBACK FRONTEND] "
        + str(recipe_to_product_mapping[recipe_path])
        + " [END CALLBACK FRONTEND]"
    )



memory = ConversationBufferMemory(memory_key="chat_history")
memory.clear()

tools = [
    retrieve_recipes,
    retrieve_products,
    get_recipe_detail,
    get_suggested_products_for_recipe,
    recipe_selector,
]
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
)



agent.run("I would like to cook some lasagne. What are the recipes available?")


agent.run("Selecting ./recipes/lasagne.txt")

agent.run("Yes, can you give me the ingredients for that recipe?")


agent.run("Can you give me the cooking instructions for that recipe?")


agent.run("Can you give me the products I can buy for this recipe?")


agent.run("Can you show me other tomatoes you have available?")

agent.run("Nice, how about carrots?")


agent.run("Thank you, that's everything!")




