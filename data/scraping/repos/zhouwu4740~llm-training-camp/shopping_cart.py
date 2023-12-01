from typing import List
import json

from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.utils.openai_functions import convert_pydantic_to_openai_function

from bookstore import BOOK_STORRE_RETRIEVER
from utils import logger


class Book(BaseModel):
    title: str
    price: float


class ShoppingCart(BaseModel):
    books: List[Book] = []

    def add_book(self, name: str, price: float):
        self.books.append(Book(title=name, price=price))
        return "已添加到购物车"

    def remove_book(self, name: str):
        for book in self.books:
            if book.title == name:
                self.books.remove(book)
        return "已帮您删除"

    def review_book_list(self):
        return json.dumps(self.books)

    def checkout(self):
        total = 0
        for book in self.books:
            total += book.price
        return str(total)


shopping_cart = ShoppingCart()


class AddToShoppingCart(BaseModel):
    """添加商品到购物车"""
    name: str
    """商品名称"""
    price: float
    """商品价格"""


class RemoveFromShoppingCart(BaseModel):
    """从购物车中移除商品"""
    name: str
    """商品名称"""


class ReviewShoppingCart(BaseModel):
    """查看购物车清单"""


class Checkout(BaseModel):
    """购物车结算"""


template = """根据客户的问题选择对应的方法去完成客户的需求，注意不要返回方法列表以外的方法，方法需要的参数可以从history和books 
information中检索出来。如果信息不全，不能直接获取到参数值可以用None替代，但不要凭空构造数值 history: {history}

books information:
{book_documents}

request:
{request}
"""

prompt = PromptTemplate.from_template(template)


# history = ChatPromptTemplate.from_messages([
#     ("human", "我想买一本 Python 入门的书籍"),
#     ("ai", "我这里有《Python in Action》，这本书比较适合入门。也不贵，只需要 51.2 元")
# ])

def retrieval_book(request: str):
    return BOOK_STORRE_RETRIEVER.get_relevant_documents(request)


def shopping_functions():
    _add = convert_pydantic_to_openai_function(AddToShoppingCart)
    _remove = convert_pydantic_to_openai_function(RemoveFromShoppingCart)
    _review = convert_pydantic_to_openai_function(ReviewShoppingCart)
    _checkout = convert_pydantic_to_openai_function(Checkout)
    return [_add, _remove, _review, _checkout]


# openai functions
SHOPPING_FUNCTIONS = shopping_functions()


def mapping_request_function(request: str, history: ChatPromptTemplate, **kwargs):
    llm = ChatOpenAI(temperature=0).bind(functions=SHOPPING_FUNCTIONS)
    # 根据用户请求，检索出相关的书籍信息
    book_documents = retrieval_book(request)
    logger.info(f"检索出相关的书籍信息, request={request}, \n{book_documents}")
    book_template = '\n-----\n'.join([document.page_content for document in book_documents])
    book_prompt = PromptTemplate.from_template(book_template)
    logger.info(f"history: {history.format()}, book_documents: {book_prompt.format()}")
    chain = RunnablePassthrough.assign(history=history) | RunnablePassthrough.assign(
        book_documents=book_prompt) | prompt | llm

    response = chain.invoke({"request": request})
    logger.info(f"调用 openai 确定 function all function. chain: {chain}, 结果: {response}")

    function_name = response.additional_kwargs["function_call"]["name"]
    argument_string = response.additional_kwargs["function_call"]["arguments"]
    arguments = json.loads(argument_string)
    return function_name, arguments


# def mapping_request_function_wrapper(params: dict):
#     return mapping_request_function(**params)


# 调用 shopping cart方法，完成用户请求
def call_function(func_name, args):
    if func_name == "AddToShoppingCart":
        return shopping_cart.add_book(**args)
    elif func_name == "RemoveFromShoppingCart":
        return shopping_cart.remove_book(**args)
    elif func_name == "ReviewShoppingCart":
        return shopping_cart.review_book_list()
    elif func_name == "Checkout":
        return shopping_cart.checkout()
    else:
        raise NotImplementedError(f"Unsupported function: {func_name}")


def finish_request(params: dict):
    logger.info(f"调用 finish_request, dict: {params}")
    function_name, arguments = mapping_request_function(**params)
    logger.info(f"调用 shopping cart function, function_name: {function_name}, arguments: {arguments}")
    result = call_function(function_name, arguments)
    return {"result": result}


if __name__ == '__main__':
    # history = ChatPromptTemplate.from_messages([
    #     ("human", "我想买一本 Python 入门的书籍"),
    #     ("ai", "我这里有《Python in Action》，这本书比较适合入门。也不贵，只需要 51.2 元")
    # ])
    # request = "我想买一本 Python 入门的书籍"
    # function_name, arguments = mapping_request_function(request, history)
    # print(function_name, arguments)
    # print(call_function(function_name, arguments))

    request = "帮我将《Python in Action》加入购物车"
    book_documents = retrieval_book(request)
    logger.info(f"检索出相关的书籍信息, request={request}, \n{book_documents}")
    book_template = '\n-----\n'.join([document.page_content for document in book_documents])
    book_prompt = PromptTemplate.from_template(book_template)
    logger.info(book_prompt)
