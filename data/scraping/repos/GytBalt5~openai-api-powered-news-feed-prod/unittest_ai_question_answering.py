from unittest.mock import patch

from django.test import TestCase

from pandas import DataFrame

from openaiapp.ai_question_answering import AbstractAIQuestionAnswering
from openaiapp.factories import (
    EmbeddingsFactory,
    TextPreparatoryFactory,
    AIQuestionAnsweringFactory,
)


class AIQuestionAnsweringTestCase(TestCase):
    def setUp(self):
        self.texts = [
            "Fact-based news, exclusive video footage, photos and updated maps. Abra kadabra abra kadabra YEAH.",
            "Fact-based news, exclusive video footage, photos and updated maps. Abra kadabra abra kadabra Csharp.",
        ]
        df = DataFrame({"text": self.texts})
        df_embeddings_object = EmbeddingsFactory().create_object(input_type=DataFrame)

        df = df_embeddings_object.create_embeddings(input=df)
        self.df = df_embeddings_object.flatten_embeddings(input=df)

        self.ai_qa = AIQuestionAnsweringFactory().create_object(
            text_embeddings_object=EmbeddingsFactory().create_object(input_type=str),
            text_preparatory=TextPreparatoryFactory().create_object(df=self.df),
        )

    def test_should_question_answering_instance_inherits_abstract(self):
        """
        Test that the AI question answering instance inherits from the AbstractAIQuestionAnswering class.
        """
        self.assertIsInstance(self.ai_qa, AbstractAIQuestionAnswering)

    def test_should_be_created_context_for_question(self):
        """
        Test that a relevant context is created for a given question by finding the most similar text from the data frame.
        """
        question = "What are the pros and cons of the Csharp programming language?"
        self.ai_qa.context_max_len = 30
        context = self.ai_qa.create_context(question=question)

        self.assertIsInstance(context, str)
        self.assertEqual(context, self.texts[1])

    def test_should_answer_question(self):
        """
        Test that the AI question answering system can provide an answer to a given question.
        """
        with patch(
            f"openaiapp.ai_question_answering.{type(self.ai_qa).__name__}.create_context"
        ) as mock_create_context, patch(
            "openai.Completion.create"
        ) as mock_completion_create:
            mock_create_context.return_value = "context"
            mock_completion_create.return_value = {
                "choices": [{"text": "I don't know"}]
            }
            question = "What are the pros and cons of Matcha tea?"
            answer = self.ai_qa.answer_question(question=question)

            self.assertIsInstance(answer, str)
            self.assertEqual(answer, "I don't know")
