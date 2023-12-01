# from langchain.utilities import GoogleSerperAPIWrapper
from langchain.utilities import SearxSearchWrapper

def load_tools():
    # search = GoogleSerperAPIWrapper()
    search = SearxSearchWrapper(searx_host="http://localhost:8080")

    def searchGoogle(key_word):        
        res = search.run(key_word)
        # print ('key_word',key_word)
        # print ('res',res)
        return res
    
    dict_tools = {
        'Google Search': searchGoogle
    }
    return dict_tools
