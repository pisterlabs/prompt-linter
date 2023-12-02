import os
import sys
sys.path.join(os.path.dirname(os.path.dirname("./")))

import unittest
from unittest.mock import MagicMock, AsyncMock
from anthropic import ACompleteResponse
from services.claude.claude_api import AnthropicAPI

class TestAnthropicAPI(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_response = ACompleteResponse(prompt="Hello, world!", completions=["Hello, world!"])
        self.mock_client.acomplete = AsyncMock(return_value=self.mock_response)
        self.api = AnthropicAPI(model="claude-v1", client=self.mock_client)

    def test_generate_text(self):
        prompt = "Hello, world!"
        params = {"temperature": 0.5}
        expected_output = self.mock_response
        result = self.api.generate_text(prompt, params)
        self.assertEqual(result, expected_output)
        self.mock_client.acomplete.assert_called_once_with(
            prompt="Hello, world!",
            model="claude-v1",
            params={"temperature": 0.5}
        )