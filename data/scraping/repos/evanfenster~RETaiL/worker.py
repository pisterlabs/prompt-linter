from typing import Optional
from langchain.chains.openai_functions import (
    create_openai_fn_chain,
    create_structured_output_chain,
)
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
import sqlite3

global chain, cursor, item_picker, connection

# ------------------------
# DATABASE FUNCTIONS
# ------------------------
def get_items() -> set:
    """Get all items in the database."""

    cursor.execute("SELECT ProductName FROM Inventory")
    items = cursor.fetchall()
    if items:
        return set([item[0] for item in items])
    else:
        return None

def get_id(item: str, first_time:bool=True) -> int:
    """Get the ProductID of an item. 

    Args:
        item: The item to get the ProductID of.
    """

    # First, try to see if the default item name is in the database
    cursor.execute("SELECT ProductID FROM Inventory WHERE ProductName = ?", (item,))
    id = cursor.fetchone()
    if id:
        return id[0]
    elif not first_time:
        return None
    else:
        # If this is our first time trying an english search, ask an LLM to try and match it to an item in the database
        items = get_items()
        if items:
            items = list(items)
            items = ", ".join(items)
            # Get the product name that matches our item name most closely, or none if it appears our item is not in the database
            name = item_picker.predict("Which item from the list matches " + item + "? Output 'None' if it is not in the list.\n" + items)
            print("NAME: ")
            print(name)
            return get_id(name, False)
        else:
            return None

def get_quantity(item: str) -> int:
    """Get the quantity of an item in stock from its ID. The first letter should be capitalized, and the item should not be pluralized.

    Args:
        item: The item to get the quantity of.
    """

    id = get_id(item)

    cursor.execute("SELECT QuantityInStock FROM Inventory WHERE ProductID = ?", (id,))
    quantity = cursor.fetchone()
    if quantity:
        return quantity[0]
    else:
        return None

def get_price(item: str) -> int:
    """Get the price of an item from its ID. The first letter should be capitalized, and the item should not be pluralized.

    Args:
        item: The item to get the price of.
    """

    id = get_id(item)

    cursor.execute("SELECT Price FROM Inventory WHERE ProductID = ?", (id,))
    price = cursor.fetchone()
    if price:
        return price[0]
    else:
        return None
    
def get_description(item: str) -> int:
    """Get the description of an item from its ID. The first letter should be capitalized, and the item should not be pluralized.

    Args:
        item: The item to get the description of.
    """

    id = get_id(item)

    cursor.execute("SELECT Description FROM Inventory WHERE ProductID = ?", (id,))
    description = cursor.fetchone()
    if description:
        return description[0]
    else:
        return None

def get_aisle(item: str) -> int:
    """Get the aisle of an item from its ID. The first letter should be capitalized, and the item should not be pluralized.

    Args:
        item: The item to get the aisle of.
    """

    id = get_id(item)

    cursor.execute("SELECT Aisle FROM Inventory WHERE ProductID = ?", (id,))
    aisle = cursor.fetchone()
    if aisle:
        return aisle[0]
    else:
        return None
    
def get_instore(item: str) -> str:
    """Get whether an item is in the store or not from its ID. The first letter should be capitalized, and the item should not be pluralized.

    Args:
        item: The item to get the in-store status of.
    """

    id = get_id(item)
    if id:
        return "True" # need to return english for the llm to understand
    else:
        return "False"
    

# ------------------------
# MAIN FUNCTIONS
# ------------------------
def setup_worker() -> Optional[ChatOpenAI]:
    # If we pass in a model explicitly, we need to make sure it supports the OpenAI function-calling API.
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    prompt_msgs = [
        SystemMessage(
            content="You are a world class algorithm for extracting information in structured formats."
        ),
        HumanMessage(
            content="Use the given format to extract information from the following input:"
        ),
        HumanMessagePromptTemplate.from_template("{input}"),
        HumanMessage(content="Tips: Make sure to answer in the correct format"),
    ]
    prompt = ChatPromptTemplate(messages=prompt_msgs)

    chain = create_openai_fn_chain([get_instore, get_quantity, get_price, get_aisle, get_description], llm, prompt, verbose=True)
    return chain

def query(question: str) -> str:
    print("QUESTION: " + question)
    details = chain.run(question)
    print("DETAILS: ")
    print(details)

    # Call the function 
    func_name = details["name"]
    args = details["arguments"]
    func = globals()[func_name]
    result = func(**args)

    if result:
        return result
    else:
        return "NA"

def main():
    global cursor, chain, item_picker, connection
    load_dotenv()

    # Connect to the SQLite database
    connection = sqlite3.connect('inventory.db')
    cursor = connection.cursor()
    
    # Get our LLM chains
    chain = setup_worker()
    item_picker = ChatOpenAI(model_name="gpt-3.5-turbo-0613", temperature=0)


if __name__ == "__main__":
    main()