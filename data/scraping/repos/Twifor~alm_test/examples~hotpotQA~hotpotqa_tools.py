from typing import Union, Dict
from agent.tools import Tool
from langchain import Wikipedia
from langchain.agents.react.base import DocstoreExplorer
import re
import string


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def EM(answer, key) -> bool:
    return normalize_answer(answer) == normalize_answer(key)


class WikiEnv:
    def __init__(self, gt_answer):
        self.explorer = DocstoreExplorer(Wikipedia())
        self.gt_answer = gt_answer

    def search(self, content):
        try:
            observation = self.explorer.search(content).strip('\n').strip()
        except Exception as e:
            print(e)
            observation = f'Could not find that page, please try again.'
        return observation

    def lookup(self, content):
        try:
            observation = self.explorer.lookup(content).strip('\n').strip()
        except ValueError:
            observation = f'The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given.'
        return observation

    def finish(self, content):
        reward = EM(content, self.gt_answer)
        if reward == True:
            return "Your answer is CORRECT.", 1
        return "Your answer is INCORRECT.", 0


class SearchTool(Tool):
    def __init__(self, wikiEnv: WikiEnv):
        super().__init__()
        self.wikiEnv = wikiEnv
        self.invoke_label = "Search"

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        return self.wikiEnv.search(invoke_data), 0, False, {}

    def description(self) -> str:
        return "Search(entity), which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search."


class LookupTool(Tool):
    def __init__(self, wikiEnv: WikiEnv):
        super().__init__()
        self.wikiEnv = wikiEnv
        self.invoke_label = "Lookup"

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        return self.wikiEnv.lookup(invoke_data), 0, False, {}

    def description(self) -> str:
        return "Lookup(keyword), which returns the next sentence containing keyword in the current passage."


class FinishTool(Tool):
    def __init__(self, wikiEnv: WikiEnv):
        super().__init__()
        self.wikiEnv = wikiEnv
        self.invoke_label = "Finish"

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        res = self.wikiEnv.finish(invoke_data)
        return res[0], res[1], True, {"answer": invoke_data, "gt_answer": self.wikiEnv.gt_answer}

    def description(self) -> str:
        return "Finish(answer), which returns the answer and finishes the task."
