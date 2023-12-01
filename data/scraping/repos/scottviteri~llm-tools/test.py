import unittest
import argparse
import subprocess
import os
import io
import json
import sys
import contextlib
from unittest.mock import patch

from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from lm_cli import main  # Import the main function from lm module

dir_path = os.path.dirname(os.path.realpath(__file__))
# Specify the relative path to the claude_history.json file
history_file = os.path.join(dir_path, 'claude_history.json')

class TestAnthropicMethods(unittest.TestCase):

    def setUp(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('-m', '--message', type=str, help='Input message')
        self.parser.add_argument('-c', '--chat', nargs='*', default=[], type=str, help='Chat IDs')
        self.parser.add_argument('-r', '--read', type=str, help='Read chat by ID')
        self.parser.add_argument('-d', '--delete', type=str, help='Delete chat by ID')
        self.parser.add_argument('-u', '--undo', type=str, help='Undo last message in chat by ID')
        self.parser.add_argument('-s', '--shell', type=str, help='Request shell command')
        self.parser.add_argument('-p', '--proglang', type=str, help='Request program in language')
        self.parser.add_argument('-l', '--list', action='store_true', help='List all chat IDs')
        self.parser.add_argument('-db', '--debug', action='store_true', help='Debug mode')

    def test_chat_creation(self):
        # Delete the chat if it exists
        sys.argv = ['lm', '--delete', 'example']
        args = self.parser.parse_args()
        main(args)

        # Create a new chat
        sys.argv = ['lm', '--message', 'Hello', '--chat', 'example']
        args = self.parser.parse_args()
        main(args)

        # Check if the chat was created
        with open(history_file, 'r') as f:
            histories = json.load(f)
        self.assertIn('example', histories)

    def test_chat_continuation(self):
        # Create a new chat
        sys.argv = ['lm', '--message', 'Hello', '--chat', 'example']
        args = self.parser.parse_args()
        main(args)

        # Continue the chat
        sys.argv = ['lm', '--message', 'How are you?', '--chat', 'example']
        args = self.parser.parse_args()
        main(args)

        # Check if the chat was continued
        with open(history_file, 'r') as f:
            histories = json.load(f)
        self.assertIn('How are you?', histories['example'])

    def test_chat_reading(self):
        # Delete the chat if it exists
        sys.argv = ['lm', '--delete', 'example']
        args = self.parser.parse_args()
        main(args)

        # Create a new chat
        sys.argv = ['lm', '--message', 'Hello', '--chat', 'example']
        args = self.parser.parse_args()
        main(args)

        # Read the chat
        sys.argv = ['lm', '--read', 'example']
        args = self.parser.parse_args()
        with patch('builtins.print') as mocked_print:
            main(args)

        # Check the print output
        call_args = mocked_print.call_args[0]
        self.assertTrue(call_args[0].startswith(" \n\nHuman: Hello \n\nAssistant:"))


    def test_undo_last_message(self):
        # Create a new chat
        sys.argv = ['lm', '--message', 'Hello', '--chat', 'example']
        args = self.parser.parse_args()
        main(args)

        # Undo the last message
        sys.argv = ['lm', '--undo', 'example']
        args = self.parser.parse_args()
        main(args)

        # Check if the last message was undone
        with open(history_file, 'r') as f:
            histories = json.load(f)
        self.assertNotIn('How are you?', histories['example'])


    def test_list_chats(self):
        # Simulate command line arguments
        sys.argv = ['lm', '--list']
        args = self.parser.parse_args()
        
        # Call the main function and capture the output
        with contextlib.redirect_stdout(io.StringIO()) as f:
            main(args)
        output = f.getvalue()
        
        # Check if the list of chats was printed
        self.assertIn('example', output)


    
    # Add more tests for the other arguments here...

