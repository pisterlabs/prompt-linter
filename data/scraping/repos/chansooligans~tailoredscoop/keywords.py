import logging
from dataclasses import dataclass

from tailoredscoop import openai_api


@dataclass
class Keywords:
    def __post_init__(self):
        self.logger = logging.getLogger("tailoredscoops.Keywords")

    def get_similar_keywords_from_gpt(self, kw):

        messages = [
            {
                "role": "system",
                "content": "Generate at least 5 similar keywords to query news articles so that it's likely to find recent results",
            },
            {"role": "system", "content": "Return keywords in comma separated format"},
            {"role": "system", "content": "Example input: 'supreme court'"},
            {
                "role": "system",
                "content": "Example output: 'SCOTUS, justice, judiciary, constitutional law, courts'",
            },
            {
                "role": "system",
                "content": "If error or there are no similar keywords, return ''",
            },
            {"role": "user", "content": f"keywords: {kw}"},
            {"role": "system", "content": "keywords:"},
        ]

        response = self.openai_api.create(
            model="gpt-3.5-turbo", messages=messages, temperature=0.2, max_tokens=10
        )["choices"][0]["message"]["content"]

        self.logger.info(f"{kw} | similar keyword: {response}")

        if response.startswith("Sorry, ") | response.startswith("I'm, "):
            return ""

        return response.replace('"', "").replace("'", "")

    def get_topic(self, kw):

        messages = [
            {"role": "system", "content": "You are a classification tool."},
            {
                "role": "system",
                "content": "Given keywords, please classify them to a common news subcategory. The subcategory must be one of: business, entertainment, healthcare, science, sports, technology, us, world, tv, books, arts, design, celebrities, environment, politics",
            },
            {
                "role": "system",
                "content": "If error or there are no similar keywords, return general",
            },
            {"role": "user", "content": f"keywords: {kw}"},
            {"role": "system", "content": "subcategory:"},
        ]

        response = self.openai_api.create(
            model="gpt-3.5-turbo", messages=messages, temperature=0.1, max_tokens=2
        )["choices"][0]["message"]["content"]

        self.logger.info(f"using topic: {kw} | got topic: {response}")

        if response.startswith("Sorry, ") | response.startswith("I'm, "):
            return "general"

        return response
