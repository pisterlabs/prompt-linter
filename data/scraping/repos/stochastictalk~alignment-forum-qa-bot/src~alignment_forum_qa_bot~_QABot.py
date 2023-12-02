from collections import defaultdict
from typing import List

import openai

from .retriever import KeywordSearchRetriever
from .retriever import RetrievedParagraph


class QABot:
    def __init__(self, openai_api_key: str = None):
        """
        Parameters
        ----------
        openai_api_key : str
            OpenAI API key.
        """
        openai.api_key = openai_api_key
        self._post_retriever = KeywordSearchRetriever()

    def query(self, query_text: str, method="summarize-first") -> str:
        """Asks a question about alignment forum's post corpus using OpenAI's completion API.

        Parameters
        ----------
        text : str
            The query input by the Alignment Forum user.
        """
        posts = self._post_retriever.retrieve(query=query_text)
        return self._get_response(query_text, posts, method=method)

    def _get_response(self, query_text: str, posts: List[RetrievedParagraph], method="summarize-first"):
        case_switch = {
            "summarize-first": self._summarize_first,
            "summarize-by-author": self._summarize_by_author,
        }

        return case_switch[method](query_text, posts)

    def _summarize_by_author(self, query_text: str, posts: RetrievedParagraph):
        posts_by_author = defaultdict(list)
        for post in posts:
            posts_by_author[post["author"]].append(post["paragraph"])
        author_post_pairs = list(posts_by_author.items())
        result = ""
        for author, posts in author_post_pairs[:5]:
            result += f"{author}'s views:\n\n"
            prompt_for_author_posts = self._get_prompt_for_author_posts(posts, query_text)
            completion = self._get_response_for_author_prompt(prompt_for_author_posts)
            result += completion + "\n\n"
        return result

    def _get_prompt_for_author_posts(self, posts: List[str], query_text: str):
        prompt = f"""
        I'm going to show you some sample paragraphs written by a user on the Alignment Forum.
        Please summarize the views of this author on {query_text} in less than 200 words.
        """
        for post in posts[:10]:
            prompt += f"\n\n{post}\n\n"
        prompt += "Answer:"
        return prompt

    def _get_response_for_author_prompt(self, author_prompt: str):
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=author_prompt,
            max_tokens=2000,
        )
        return response["choices"][0]["text"]

    def _summarize_first(self, query_text: str, posts: RetrievedParagraph):
        """Maps posts and question to prompt."""

        try:
            prompt = f"""
            Summarize the following text in less than 200 words.

            {posts[0]["paragraph"]}
            """
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                # temperature=0.8,
                max_tokens=2000,
                # top_p=1,
                # frequency_penalty=0,
                # presence_penalty=0,
                # stop=["\n"],
            )
            return response["choices"][0]["text"]
        except IndexError:
            return f"No posts were found that mention the term '{query_text}'."
