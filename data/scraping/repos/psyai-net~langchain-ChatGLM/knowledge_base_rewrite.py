import pydantic
from fastapi import Body
from configs.model_config import (llm_model_dict, LLM_MODEL, REWRITE_PROMPT_TEMPLATE,
                                  VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD)
from server.utils import BaseResponse
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from typing import List
from langchain.prompts.chat import ChatPromptTemplate
from server.chat.utils import History
from server.knowledge_base.kb_service.base import KBServiceFactory
from server.knowledge_base.kb_doc_api import search_docs



class RewriteResponse(BaseResponse):
    response: List[str] = pydantic.Field(..., description="List of names")

    class Config:
        schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
                "response": ["text1", "text2", "text3"],
            }
        }

async def knowledge_base_rewrite(query: str = Body(..., description="用户输入", examples=["你好"]),
                        knowledge_base_id: str = Body(..., description="知识库名称", examples=["kb1"]),
                        topics: List[str] = Body([], 
                                                 description="内容主题",
                                                 examples=["介绍商品的价格", "介绍商品的功能", "介绍商品的使用方法"]),                        
                        ):
    kb = KBServiceFactory.get_service_by_name(knowledge_base_id)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_id}")


    model = ChatOpenAI(
        streaming=False,
        verbose=True,
        openai_api_key=llm_model_dict[LLM_MODEL]["api_key"],
        openai_api_base=llm_model_dict[LLM_MODEL]["api_base_url"],
        model_name=LLM_MODEL
    )
    top_k = VECTOR_SEARCH_TOP_K;
    score_threshold = SCORE_THRESHOLD;

    history = []
    result_data = []

    for topic in topics:
        docs = search_docs(topic, knowledge_base_id, top_k, score_threshold)
        context = "\n".join([doc.page_content for doc in docs])
        
        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_tuple() for i in history] + [("human", REWRITE_PROMPT_TEMPLATE)])

        chain = LLMChain(prompt=chat_prompt, llm=model)

        result = await chain.acall({"context": context, "query": query, "topic" : topic})                

        history.append(History(**{"role":"user","content":result["query"]}))
        history.append(History(**{"role":"user","content":result["topic"]}))
        history.append(History(**{"role":"assistant","content":result["text"]}))                

        result_data.append(result["text"])
    
    return RewriteResponse(response=result_data, code=200, msg="success")

