import json
import requests
from langchain.agents import Tool

def github_search(query):
    url = "https://api.github.com/search/repositories"
    headers = {"Accept": "application/vnd.github+json"}
    params = {"q": query, "sort": "stars", "order": "desc", "per_page": 5}
    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        print("Request successful")
        data = response.json()
        repos = data["items"]

        if not repos:
            return "No repo found, rewrite your query"

        result = ""

        for repo in repos:
            name_with_owner = repo.get("full_name", "")
            description = repo.get("description", "")
            stars = repo.get("stargazers_count", 0)
            forks = repo.get("forks_count", 0)
            url = repo.get("html_url", "")

            result += f"{name_with_owner}\n{description}\nStars: {stars}, Forks: {forks}\n{url}\n\n"

        return result.strip()
    else:
        print("Request failed")
        return f"Error: {response.text}"

github_search_tool = Tool(
    name="GitHub Search",
    func=github_search,
    description="A tool that searches for GitHub repositories based on a query."
)

query = "impedance control "
output = github_search_tool(query)
print("Output:")
print(output)
