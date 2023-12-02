import os
import json
import openai

from dataclasses import dataclass

openai.api_key = os.getenv("OPENAI_API_KEY")

@dataclass
class ArticleNPK: # NPK = No Primary Key, See database schema
    title: str
    nofield_abstract: str # Does not appear in the database, just to give the AI more context
    date_written: str | None
    by_line: str
    header_image_url: str | None
    header_image_attribution: str | None
    source_domain: str
    sorted_categories: dict[str, list[str]]
    original_url: str
    content: str # Can be "<< not yet generated >>"
    skip_cheesify: bool = False

    def generate_cheesy_title_prompt(self):
        return f"Title: {self.title}\nAbstract: {self.nofield_abstract}\nCategories: {self.sorted_categories}"
    
    def generate_cheesy_title(self):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": """
                    I will give you information about a news article.
                    You will rewrite the title to be about cheese, but ensure it is about the same topic.
                    It must reference the categories given.
                    No matter what, respond with only the title.
                    Do not add any other information or formatting.
                    Do not respond with categories.
                    """.strip(),
                },
                {
                "role": "user",
                "content": self.generate_cheesy_title_prompt(),
                }
            ],
            temperature=1,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        self.title = response["choices"][0]["message"]["content"]
    
    def generate_content_prompt(self):
        return f"Title: {self.title}\nCategories: {self.sorted_categories}"
    
    def generate_content(self):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": """
                    I will give the title and some other information about a news article.
                    Write the article.
                    It should be about cheese, but absolutely must also reference the categories given.
                    No matter what, respond with only the article content.
                    Do not add any other information or formatting.
                    """.strip(),
                },
                {
                "role": "user",
                "content": self.generate_content_prompt(),
                }
            ],
            temperature=1,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        self.content = response["choices"][0]["message"]["content"]
    
    def fix_title(self):
        for tstr in ("title:", "title: "):
            if self.title.lower().startswith(tstr):
                self.title = self.title[len(tstr):]
        self.title = self.title.split('\n')[0]

    def cheesify(self):
        if self.skip_cheesify:
            return
        print("[[t", end="", flush=True)
        self.generate_cheesy_title()
        print("]", end="", flush=True)
        self.fix_title()
        print("][c", end="", flush=True)
        self.generate_content()
        print("]", end="", flush=True)
