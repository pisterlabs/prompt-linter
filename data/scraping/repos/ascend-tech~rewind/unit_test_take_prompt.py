import unittest
from unittest.mock import MagicMock, patch
from flask import Flask, request
from io import BytesIO
import sys
sys.path.append('C:\\Users\\Smruti\\Desktop\\rewind\\ai\\files')
from voicebot import take_prompt
from bson.objectid import ObjectId

class TestTakePrompt(unittest.TestCase):

    @patch("voicebot.openai.Audio.transcribe")
    @patch("voicebot.collection.find_one")
    def test_take_prompt(self, mock_collection, mock_transcribe):
        
        request.user_id = 12345
        # Set up mock response from OpenAI API
        # Create a mock audio file to be uploaded
        audio_data = "this is testing".encode("utf-8")
        audio_file = MagicMock()
        audio_file.read.return_value = audio_data
        audio_file.content_type = "audio/wav"
        audio_file.filename = "test.wav"
        
        app = Flask(__name__)
        app.config["TESTING"] = True
        
        # Create a mock request object
        with app.test_request_context("/", method="GET", data={"audio": audio_file}):
            
            # Call the function and get the response
            response = take_prompt()
            
            # Assert that the response is correct
            self.assertEqual(response, "this is testing")
            
            # Assert that the status was updated in the database
            self.assertEqual(mock_collection.find_one({"user_id": 12345})["status"], "transcribed")
            self.assertEqual(mock_collection.find_one({"user_id": 12345})["prompt"], "This is a test transcript.")
            
            # Assert that the OpenAI API was called with the correct parameters
            mock_transcribe.assert_called_with("whisper-1", audio_file, models=["en"], response_format="text")

if __name__ == "__main__":
    unittest.main()
