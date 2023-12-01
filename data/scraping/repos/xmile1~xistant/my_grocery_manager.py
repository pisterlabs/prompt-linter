from copy import deepcopy
from functools import partial
from hmac import new
from http import cookies
import json
import re
from typing import Any
import aiohttp
from numpy import product
import requests
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.agents.tools import Tool
from langchain.tools import BaseTool
from langchain.agents.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain.requests import RequestsWrapper

from langchain.chat_models import ChatOpenAI
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents import AgentExecutor, ConversationalAgent
from config import settings


class GroceryManagerPlugin():
  def __init__(self, model):
      self.model = model
      self.headers = {
        "GROCY-API-KEY": settings.grocy_api_key
      }

      tools = [
        self.get_product_id_tool(),
        self.get_stock_item_tool(),
        self.add_purchased_items_tool(),
      ]

      agent = ZeroShotAgent.from_llm_and_tools(
          llm=self.model,
           tools=tools,
           verbose=True,
      )

      
      self.agent = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True
      )

  def get_lang_chain_tool(self):
    
    return [
      Tool(
          name="Grocery Manager",
          description="The Grocery Manager is a tool for managing your grocery stock. With this tool, you can easily retrieve, create, update, and delete data related to your grocery items. the input must be the initial user request",
          func= self.agent.run,
          return_direct=True,
      )]
     

  def get_product_id_tool(self):
    def get_product_id(input: str) -> Any:
      products = requests.get(settings.grocy_api_url + "/objects/products", headers=self.headers).json()
      all_names_with_id_as_text = "\n".join([f"{p['name']}: {p['id']}" for p in products])

      return all_names_with_id_as_text



    return Tool(
      name="Get Product ID",
      description="This tool returns all the product names with their ids, you MUST use this tool before any other tool",
      func=get_product_id,
    )
  
  def get_stock_item_tool(self):
    def get_product_stock_details(input: str) -> str:
      product_ids = input.split(",")
      product_ids = [p.strip() for p in product_ids]

      products = []
      for product_id in product_ids:
        product = requests.get(settings.grocy_api_url + f"/stock/products/{product_id}", headers=self.headers).json()
        products.append(product)
      return json.dumps(products)

    return Tool(
      name="Get a single product's details",
      description="""This tool returns the details for a product or list of products including the stock details, the input is the product id of the item. The stock amount represents the quantity we have available. the input must be product ids separated by commas, for example: 1, 2, 3""",
      func=get_product_stock_details,
    )
  
  def add_purchased_items_tool(self):
    def add_purchased_items(input: str) -> str:
      products_to_add = json.loads(input)
      products = []
      for product in products_to_add:
        data = {
          # remove negative sign from amount
          "amount": abs(product["amount"]),
          "transaction_type": product["transaction_type"],
        }
        path_suffix = "add" if product["transaction_type"] == "purchase" else "consume"
        product = requests.post(settings.grocy_api_url + f"/stock/products/{product['product_id']}/{path_suffix}", headers=self.headers, data=data).json()
        products.append(product)
      return json.dumps(products)

    return Tool(
      name="Add or Consume stock items",
      description="This tool adds or consumes stock items in the database, it requires a product id and the amount to add or remove from stock and a transaction type that is one of add or consume, for example: [{{\"product_id\": 1, \"amount\": 1, transaction_type: \"purchase\"}}]. you must retrieve the product id using the get product id tool",
      func=add_purchased_items,
    )

