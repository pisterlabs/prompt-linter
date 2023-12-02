from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import boto3
from typing import Any, Dict, List, Optional
from langchain.schema import BaseRetriever, Document
from langchain.docstore.document import Document
import re



def clean_result(res_text):
    res = re.sub("\s+", " ", res_text).replace("...","")
    return res
    
def get_top_k_results(resp, count):
    r = resp["ResultItems"][count]
    doc_title = r["DocumentTitle"]["Text"]
    doc_uri = r["DocumentURI"]
    r_type = r["Type"]
    if (r["AdditionalAttributes"] and r["AdditionalAttributes"][0]["Key"] == "AnswerText"):
        res_text = r["AdditionalAttributes"][0]["Value"]["TextWithHighlightsValue"]["Text"]
    else:
        res_text = r["DocumentExcerpt"]["Text"]
    doc_excerpt = clean_result(res_text)
    page_no = ''
    
    if r["DocumentAttributes"]:
        for da in r['DocumentAttributes']:
            if da['Key'] == '_excerpt_page_number':
                page_no = str(da["Value"]["LongValue"])

    combined_text = "Document Title: " + doc_title + "\nDocument Excerpt: \n" + doc_excerpt + "\n"
    return {"page_content":combined_text, "metadata":{"source":doc_uri, "title": doc_title, "excerpt": doc_excerpt, "type": r_type,'page':page_no}}

def kendra_query(client,query,top_k,index_id,filter_key=None,filter_value=None):
    af = {}
    if filter_key and filter_key.strip() !="" and filter_value and filter_value.strip() !="" :
        af = {
            "EqualsTo": {      
                "Key": filter_key,
                "Value": {
                    "StringValue": filter_value
                    }
                }
              }
    
    response = client.query(IndexId=index_id, QueryText=query.strip(),AttributeFilter=af)
    if len(response["ResultItems"]) > top_k:
        r_count = top_k
    else:
        r_count = len(response["ResultItems"])
    docs = [get_top_k_results(response, i) for i in range(0, r_count)]
    return [Document(page_content = d["page_content"], metadata = d["metadata"]) for d in docs]

def kendra_client(region):
    client = boto3.client('kendra', region_name=region)
    return client


class KendraIndexRetriever(BaseRetriever):
    index_id: str
    """Kendra index id"""
    region: str
    """AWS region of the Kendra index"""
    top_k: int
    """Number of documents to query for."""
    return_source_documents: bool
    """Whether source documents to be returned """
    client: Any
    """ boto3 client for Kendra. """
    filter_key: str
    """Filter key for Kendra"""
    filter_value: str
    """Filter value for Kendra"""
    
    
    def __init__(self, index_id,region,top_k=3,return_source_documents=True,filter_key=None,filter_value=None):
        super().__init__(index_id=index_id,region=region,top_k=top_k,return_source_documents=return_source_documents,filter_key=filter_key,filter_value=filter_value,client=None)
        self.index_id = index_id
        self.region = region
        self.top_k = top_k
        self.return_source_documents = return_source_documents
        self.client = kendra_client(self.region)
        self.filter_key = filter_key
        self.filter_value = filter_value
        
    def get_relevant_documents(self, query: str) -> List[Document]:
        docs = kendra_query(self.client, query,self.top_k, self.index_id,self.filter_key,self.filter_value)
        return docs
    
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return await super().aget_relevant_documents(query)

def build_kendra_retriever(region,index_id,filter_key=None,filter_value=None):
    return KendraIndexRetriever(index_id=index_id,region=region,return_source_documents=True,filter_key=filter_key,filter_value = filter_value)