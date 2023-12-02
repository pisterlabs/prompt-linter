import json
from typing import List

import openai
import pandas as pd
from django.utils.functional import cached_property


class Question:
    def __init__(self, **kwargs: dict):
        self.data = kwargs
        self.id = kwargs.get("Question ID")
        self.question_short = kwargs.get("Question_short")
        self.question_original = kwargs.get("Question_original")
        self.keywords = kwargs.get("Keywords")
        self.answer_plain_text = kwargs.get("Answer_plain_text")
        self.answer_orignal = kwargs.get("Answer_original")
        self.question_original_alternatives = kwargs.get(
            "Question_original_alternatives", []
        )
        self.question_short_alternatives = kwargs.get("Question_short_alternatives", [])
        self.notes = kwargs.get("Notes")
        self.source_type = kwargs.get("Source_type")
        self.date = kwargs.get("date")

    @property
    def question_set(self) -> set:
        questions = (
            [
                self.question_short,
                self.question_original,
            ]
            + self.question_original_alternatives
            + self.question_short_alternatives
        )

        return set(question for question in questions if question)

    @cached_property
    def embedding_set(self) -> List[List[float]]:
        response = openai.Embedding.create(
            input=list(self.question_set), model="text-embedding-ada-002"
        )

        return [q["embedding"] for q in response["data"]]

    @cached_property
    def dataframe(self):
        return pd.DataFrame(
            data={
                "id": [self.id] * len(self.question_set),
                "embeddings": self.embedding_set,
            }
        )

    @property
    def short_context_information(self):
        """Used for get_relevant_documents in retriever.py"""
        return json.dumps(
            {
                "question_original": self.question_original,
                "question_short": self.question_short,
                "answer": self.answer_plain_text,
                "notes": self.notes,
            },
            indent=4,
        )
