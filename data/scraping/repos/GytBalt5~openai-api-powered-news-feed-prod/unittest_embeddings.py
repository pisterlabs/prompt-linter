from typing import List

from django.test import TestCase

import numpy as np
from pandas import DataFrame

from openaiapp.embeddings import AbstractEmbeddings
from openaiapp.factories import EmbeddingsFactory


class EmbeddingsTestCase(TestCase):
    def setUp(self):
        """
        Set up the test case with sample texts, a DataFrame, and an embeddings object.
        """
        self.texts = [
            "Fact-based news, exclusive video footage, photos and updated maps. Abra kadabra abra kadabra YEAH.",
            "Fact-based news, exclusive video footage, photos and updated maps. Abra kadabra abra kadabra YEAH.",
        ]
        self.df = DataFrame({"text": self.texts})
        self.factory = EmbeddingsFactory()
        self.embeddings_obj = self.factory.create_object(input_type=DataFrame)

    def test_should_embeddings_instances_inherit_abstract(self):
        """
        Test that the embeddings instances created by the factory inherit from AbstractEmbeddings.
        """
        obj = self.factory.create_object(input_type=DataFrame)
        self.assertIsInstance(obj, AbstractEmbeddings)
        obj = self.factory.create_object(input_type=str)
        self.assertIsInstance(obj, AbstractEmbeddings)

    def test_should_raise_exception_with_unsupported_input_type(self):
        """
        Test that creating an embeddings object with an unsupported input type raises a TypeError.
        """
        with self.assertRaises(TypeError):
            self.factory.create_object(input_type=List)

    def test_should_create_embedding_for_each_text_chunk(self):
        """
        Test that an embedding is created for each text chunk in the DataFrame.
        """
        df = self.embeddings_obj.create_embeddings(input=self.df)

        self.assertIsInstance(df, DataFrame)
        self.assertEqual(set(df.columns), {"text", "embeddings"})

        text_list = df["text"].tolist()
        embedding_list = df["embeddings"].tolist()

        self.assertEqual(len(self.texts), len(embedding_list))
        self.assertIsInstance(text_list, list)
        self.assertIsInstance(embedding_list, list)
        self.assertIsInstance(text_list[0], str)
        for embedding in embedding_list:
            self.assertIsInstance(embedding, list)
            self.assertIsInstance(embedding[0], float)
            self.assertGreater(len(embedding), 0)

        self.assertEqual(self.texts, text_list)

    def test_should_flatten_embeddings_to_1d(self):
        """
        Test that embeddings are correctly flattened to a 1D numpy array.
        """
        df = self.embeddings_obj.create_embeddings(input=self.df)
        df = self.embeddings_obj.flatten_embeddings(input=df)

        self.assertIsInstance(df, DataFrame)
        self.assertEqual(set(df.columns), {"text", "embeddings"})

        text_list = df["text"].tolist()
        embedding_list = df["embeddings"].tolist()

        for embedding in embedding_list:
            self.assertIsInstance(embedding, np.ndarray)
            self.assertEqual(embedding.ndim, 1)

        self.assertIsInstance(text_list, list)
        self.assertIsInstance(embedding_list, list)
        self.assertIsInstance(text_list[0], str)
        for embedding in embedding_list:
            self.assertIsInstance(embedding[0], float)
            self.assertGreater(len(embedding), 0)

        self.assertEqual(self.texts, text_list)

    def test_should_save_flatten_embeddings_to_vector_db(self):
        """
        Test that flattened embeddings are correctly saved to the vector database.
        """
        raise NotImplementedError
