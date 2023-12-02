from django.test import TestCase

from openaiapp.factories import TokenizerFactory


class TokenizerTestCase(TestCase):
    def setUp(self):
        """
        Set up the test case with sample text and expected tokens.
        """
        self.sample_text = (
            "Fact-based news, exclusive video footage, photos and updated maps."
        )
        self.expected_tokens = [
            17873,
            6108,
            3754,
            11,
            14079,
            2835,
            22609,
            11,
            7397,
            323,
            6177,
            14370,
            13,
        ]
        self.tokenizer = TokenizerFactory().create_object()

    def test_tokenize_text(self):
        """
        Ensure that the tokenizer correctly breaks down the text into a list of tokens.
        """
        tokens = self.tokenizer.tokenize_text(self.sample_text)

        self.assertIsInstance(tokens, list, "Tokenized output should be a list.")
        self.assertEqual(
            self.expected_tokens,
            tokens,
            "Tokenized output does not match expected tokens.",
        )

    def test_decode_tokens(self):
        """
        Verify that the tokenizer accurately decodes a list of tokens back into the original text.
        """
        decoded_text = self.tokenizer.decode_tokens(tokens=self.expected_tokens)

        self.assertIsInstance(decoded_text, str, "Decoded output should be a string.")
        self.assertEqual(
            self.sample_text,
            decoded_text,
            "Decoded text does not match the original sample text.",
        )
