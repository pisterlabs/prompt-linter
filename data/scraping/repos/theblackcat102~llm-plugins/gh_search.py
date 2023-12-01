import re
import base64
from typing import Any
import requests
from llmplugins.resources import openai_enc


class GithubSearch:
    cleaning_pattern = r"\[\!\[(?!Documentation).*?\]\(.*?\)\]\(.*?\)"
    url = "https://api.github.com/search/repositories"

    prompt = """Github search: a function which allows you to search through github and reference based on the result.
    function: GithubSearch(query)
      parameters :
        - query: a string of query text
      returns: a list of github repository with name, description and readme snippet"""

    def __init__(self, gh_token) -> None:
        self.prompt = self.prompt.replace("    ", "")
        assert "github_pat_" == gh_token[:11]
        self.gh_token = gh_token

    def __call__(self, query, result_size=13, pretty_str=True) -> Any:
        res = requests.get(
            GithubSearch.url,
            params={"q": query},
            headers={
                "Authorization": "Bearer " + self.gh_token,
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        items = res.json()["items"]
        outputs = []
        for item in items[:result_size]:
            url = item["url"] + "/contents/README.md"
            res = requests.get(
                url,
                headers={
                    "Authorization": "Bearer " + self.gh_token,
                    "X-GitHub-Api-Version": "2022-11-28",
                },
            )
            readme_result = res.json()
            if "name" in readme_result:
                try:
                    readme = base64.b64decode(readme_result["content"]).decode("utf-8")
                    # remove markdown style badge
                    cleaned_readme = re.sub(GithubSearch.cleaning_pattern, "", readme)
                    outputs.append(
                        {
                            "gh_name": item["full_name"],
                            "description": item["description"],
                            "readme": cleaned_readme[:300].replace("\n\n", "\n"),
                            "issues": item["open_issues_count"],
                            "stars": item["stargazers_count"],
                            "forks": item["forks_count"],
                        }
                    )
                except UnicodeDecodeError as e:
                    continue

        if not pretty_str:
            return outputs
        output_str = ""
        for idx, item in enumerate(outputs):
            output_str += f"<|{idx+1}|> NAME: {item['gh_name']}\nstars: {item['stars']}, forks: {item['forks']}, issues: {item['issues']}\nDESCRIPTION: {item['description']}\nREADME: {item['readme']}\n---------\n"
            len_tokens = len(openai_enc.encode(output_str))
            if len_tokens > 1000:
                break

        return output_str
