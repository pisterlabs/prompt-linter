from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Dict
from langchain.output_parsers import PydanticOutputParser


def save_conversation(dict):
    memory = dict['memory']
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S") # datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")하니 할당 전에 사용됐단 오류 발생
    customer_message = dict["customer_message"]
    ai_response = f"{dict['ai_response']} {date}"
    memory.save_context({"inputs": customer_message}, {"output": ai_response})
    return ai_response


def load_memory(dict):
    memory = dict['memory']
    chat_history = memory.load_memory_variables({})['history']
    return chat_history


class Order(BaseModel):
    products: List[Dict[str, Dict[str, int]]] = Field(
        description="주문 상품 별 가격과 주문 수량\n예시: [{'상품명': {'가격': 1000, '수량': 2}}, {'상품명': {'가격': 2000, '수량': 3}}]\n모든 dictionary의 key는 예시와 동일해야만 함"
    )
    datetime: str = Field(description="현재 시간: 뒷부분에 표시된 datetime 형태의 문자열")
    
order_record_parser = PydanticOutputParser(pydantic_object=Order)