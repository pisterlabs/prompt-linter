from langchain.utilities.google_serper import GoogleSerperAPIWrapper
import os
from openai_util.function_call.openaifunc_decorator import openai_func


@openai_func
def query_info_from_google(query: str):
    """
    query info form google
    :param query: the query keyword
    """
    search = GoogleSerperAPIWrapper(serper_api_key=os.getenv('SERPER_API_KEY'), gl='cn', hl='zh-cn')
    return search.run(query)
