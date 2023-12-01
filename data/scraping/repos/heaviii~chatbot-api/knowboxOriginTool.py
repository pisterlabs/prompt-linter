from config import Config
from vectordb.ChromaDb import ChromaDb
from langchain.agents import Tool
from langchain.tools import BaseTool
from langchain.utilities import GoogleSearchAPIWrapper
from pydantic import BaseModel, Extra, root_validator


class KnowboxOriginTool(BaseTool):
    name = "KnowboxOriginTool"
    description = "useful for when you need to answer questions about the 小盒 or 花园. Input should be a fully formed question."
    #return_direct = True  # 直接返回结果

    def _run(self, query: str) -> str:
        """使用工具。"""
        print("useKnowboxTool:---",query)
        #return search.run(query)
        dbClient = ChromaDb().get_client(index="knowbox", doc_path=Config.DATA_PATH)#, reload=True
        # search_type('similarity', 'similarity_score_threshold', 'mmr')
        response = ChromaDb().retriever_search(client=dbClient, query=query, k=1, search_type="similarity")
        
        print("response:---",response)
        #print("response:---",response[0].page_content)
        #如歌没有找到，就返回空字符串，如果找到了，就返回第一个
        if len(response)==0:
            return ""
        #return str(response)
        return response[0].page_content
    
    async def _arun(self, query: str) -> str:
        """异步使用工具。"""
        print("KnowboxTool不支持异步:---",query)
        raise NotImplementedError("KnowboxTool不支持异步")
    
    def init_tool_db(self):
        db = ChromaDb().get_client("knowbox", Config.DATA_PATH)
        return db
    
    
    