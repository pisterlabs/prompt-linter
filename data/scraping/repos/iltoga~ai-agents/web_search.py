from langchain.tools.ddg_search.tool import DuckDuckGoSearchTool
from pyagents import CosineSimilarity

class WebSearchTool:
    def __init__(self):
        self.search_tool = DuckDuckGoSearchTool()
        
    def run(self, query):
        return self.search_tool.run(query)
    
    # FIXME: this model is not working. better use a different one capable of summarizing
    def summarize(self, corpus, query):
        query_matcher = CosineSimilarity()
        res = query_matcher.semantic_search(corpus=corpus, query=query, top_k=1)        
        return res

if __name__ == "__main__":
    search = WebSearchTool()
    res = search.run("Who is the president of United States?")
    print(res)