from langchain.agents import initialize_agent, AgentType, Tool
from langchain import SerpAPIWrapper
from langchain.chat_models import ChatOpenAI
import requests
import json
import os
from metaphor_python import Metaphor
from bs4 import BeautifulSoup
import re

metaphor = Metaphor(os.getenv("METAPHOR_API_KEY"))

class Functions:
    @staticmethod
    def search_papers(subject):
        """Get search results for a query"""
        try:
            search_response = metaphor.search(
                subject, num_results=3, include_domains=["arxiv.org"], use_autoprompt=True,
            )
            res = {"results":[]}
            for result in search_response.results:
                res["results"].append({"title": result.title, "url": result.url, "published_data": result.published_date, "author": result.author, "id":result.id})
            return json.dumps(res)
        except Exception as e:
            print(f"Error getting search results: {e}")
            return json.dumps({"error": "Failed to get search results"})
    
    @staticmethod
    def get_detailed_information(id):
        """Get detailed information for a query"""
        try:
            detailed_response = metaphor.get_contents([id])
            res = {"results":[]}
            for content in detailed_response.contents:
                extract = BeautifulSoup(content.extract).get_text()
                extract = re.sub("\n", ' ', extract)
                res["results"].append({"title": content.title, "content": extract, "url": content.url, "id": content.id})
            return json.dumps(res)
        except Exception as e:
            print(f"Error getting detailed results: {e}")
            return json.dumps({"error": "Failed to get search results"})
        
    @staticmethod
    def recommend_similar_resources(url):
        """Get similar results for a query"""
        try:
            similar_response = metaphor.find_similar(
                url,
                num_results=3,
            )
            res = {"results":[]}
            for result in similar_response.results:
                res["results"].append({"title": result.title, "url": result.url, "published_data": result.published_date, "author": result.author, "id": result.id})
            return json.dumps(res)
        except Exception as e:
            print(f"Error getting similar results: {e}")
            return json.dumps({"error": "Failed to get similar results"})
