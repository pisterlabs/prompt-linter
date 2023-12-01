""" Module for langchain tools
"""
from langchain.tools import Tool

import json
import logging

from  source.agents.agentv2.database.client import DatabaseClient
from  source.agents.agentv2.order import Order
import source.audio.transcriber as transcriber

import tempfile
import queue
import sys
import asyncio
import sounddevice as sd
import soundfile as sf

logger = logging.getLogger(__name__)

db = DatabaseClient()
square_order = Order()

def validate_and_search(input_str: str):
    """
    function that verifies if the requested item is in the menu and returns similar items

    Args:
        input_str (string): requested item
    """
    text_json = {}

    # convert json string to dictionary
    try: 
        text_json = json.loads(input_str)
    except:
        text_json = {"item_name" : input_str}

    if text_json == None:
        return "Cannot find the item you requested."    

    # make sure theres content.
    if len(text_json) == 0:
        return "Nothing to order in order_string."
    
    vec_sim = [] 

    # loop through items in dictionary and append to db query
    for x in text_json.values():
        temp = x
        if isinstance(x, str):
            vec_sim.append(db.query(x))
        elif "item_name" in temp.keys():
            # append the similar items found in the database
            vec_sim.append(db.query(temp["item_name"]))
    return vec_sim

def get_item_details(item_string: str) -> str:
    """When people are wondering about details of their dish

    Args:
        item_string: item to query details from

    Returns: 
        item details in text string or resulting error
    """
        
    if item_string == None:
        return "Did not get any input from the user. Prompt the user to try again."

    #ensure item exists
    vector_similarities = validate_and_search(item_string)
    print(vector_similarities)
    # if there was nothing similar
    if len(vector_similarities) == 0 or vector_similarities == [[]]:
        return "there were no dishes similar to the ones you requested."
    else:
        return vector_similarities[0][0]['description']

# used in OrderTool
def order(
        order_string: str,
    ) -> str:
    """ order tool called to place and update orders.
    
    the order_string dict can be verbose. From testing, the general item ordered is
    correct 99% of the time, but potentially when distance >= 1, we should add
    human-in-the-loop to verify or clarify that certain item.
    I don't think the other orders should fail, so add HITL after placing orders. 
    """

    # try convert the input JSON string to a dict.
    try: 
        text_json = json.loads(order_string)
    except:
        logger.error("the structured input to OrderTool was not valid JSON.")
        return "order_string was not valid JSON."
    
    if text_json == None or order_string == None or len(text_json) == 0:
        return "Did not get any input from the user. Prompt the user to try again."
    
    # vec_similarities will hold the results of a vector search for each item requested in order_string
    vec_similarities = []
    failed_orders = []
    succeeded_orders = []

    for i in range(len(text_json)):
        # empty note by default 
        note = ""
        # LLM hopefully correctly formatted and created a "order_note" field.
        if "order_note" in text_json[i].keys():
            # Occasionally the LLM likes to create the "order_note" field 
            # with a value of `null`, json.loads() converts null to NoneType
            # we dont want this b/c converting NoneType to str throws errors.
            if text_json[i]["order_note"] != None:
                note = text_json[i]["order_note"]
        
        # this is the string we are going to query the vecDB with.
        search_string = text_json[i]["item_name"]
        query_results = db.query(search_string)

        # save failed order if order fails
        if query_results == []:
            failed_orders.append([{'name' : search_string, 'error_type' : "MISSING_ITEM"}])
        elif len(query_results) == 2 and query_results[0]['_additional']['score'] == query_results[1]['_additional']['score'] and query_results[0]['name'] != query_results[1]['name']:
            query_results[0]['error_type'] = "SCORE_MATCH"
            failed_orders.append(query_results)
      
        else:
            # create initial order
            if not square_order.order_ongoing:
                success = square_order.create_order(
                    query_results[0]['item_id'], # the [0] index is bc we want the most similar vector.
                    text_json[i]["quantity_ordered"],
                    note
                )
            # add additional items to order
            else:
                success = square_order.add_item_to_order(
                    query_results[0]['item_id'],
                    text_json[i]["quantity_ordered"],
                    note
                )
            # save failed orders
            if not success:
                query_results[0]['error_type'] = "SQUARE_CALL"
                failed_orders.append(query_results)
            # order success
            else:
                succeeded_orders.append(query_results[0]["name"])   

    # nice output to user
    if len(succeeded_orders) == 0 and len(failed_orders) == 0:
        return "Nothing was picked up. Please try ordering again."
    elif len(succeeded_orders) > 0 and len(failed_orders) == 0:
        return f"""The orders for {', '.join(succeeded_orders)} succeeded! Your total is now ${(square_order.get_order_total() / 100):.2f}.""" 
    else:
        if len(succeeded_orders) > 0:
            return f"""Your orders for {', '.join(succeeded_orders)}, succeeded. Your orders for {', '.join([x[0]['name'] for x in failed_orders])} failed, and we are going to ask for some clarification. Failed Orders: {json.dumps(failed_orders)}"""
        else:
            return f"""Your orders for {', '.join([x[0]['name'] for x in failed_orders])} failed, and we are going to ask for some clarification. Failed Orders: {json.dumps(failed_orders)}"""

def get_menu(text) -> str:
    """
    Tool for retrieving the menu.

    Returns:
        Formatted string containing entire Square menu
    """
    nice_menu = []

    # parse through items in square menu and add to returning string 
    for obj in square_order.menu["objects"]:
        for variation in obj["item_data"]["variations"]:
            temp_item = ((variation["item_variation_data"]["name"] + " " + obj["item_data"]["name"]).lower().replace(",", "").replace("(", "").replace(")", ""))
            nice_menu.append(temp_item)
    return f"""***RESTAURANT MENU*** {nice_menu}"""
    
def no_order(text) -> str:
    """Tool for when the user doesn't order anything."""
    return f"""You didn't order anything. Try again?"""
    
def get_order_items(text) -> str:
    """Tool for getting the user's ordered items."""
    return f""" You have ordered the following: {", ".join(square_order.get_order_items())}, with a total of ${(square_order.get_order_total() / 100):.2f}."""


def checkout(text) -> str:
    success = square_order.start_checkout()
    if success:
        return "Thanks for dining with us!"
    else:
        return "Checkout failed."

# Below are all the tools.
OrderTool = Tool(
    name = "order_tool",
    description = "This tool is for creating new orders or adding items to existing orders.",
    return_direct=True, # dont want order tool to get called.
    func=order,
)

DescriptionTool = Tool(
    name = "get_item_description",
    description = "This tool is for describing items to customers.",
    return_direct=True, # dont want order tool to get called.
    func=get_item_details,
)

MenuTool = Tool(
    name = "get_menu_tool",
    description = "This tool is to get a restaurant menu for customers",
    return_direct = True,
    func = get_menu,
)

NoOrderTool = Tool(
    name = "no_order_tool",
    description = "when the user doesn't order anything, or there is no input.",
    return_direct = True,
    func = no_order,
)

GetUserOrderTool = Tool(
    name = "get_user_order_tool",
    description = "When the user asks about what is in their order. Do not hallucinate.",
    return_direct = True,
    func = get_order_items,
)

OrderCheckoutTool = Tool(
    name="make_order_checkout",
    description="""For when the customer requests to checkout their order.**IMPORTANT** Only call this when the customer is asking to check out.""",
    return_direct=True,
    func=checkout
)
