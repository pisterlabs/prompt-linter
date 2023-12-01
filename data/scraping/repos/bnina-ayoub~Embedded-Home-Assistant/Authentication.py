import openai
import os
# Set your OpenAI API key here
def read_api_key():
  """Reads the API key from a file."""
  api_key_file = os.path.join(os.path.dirname(__file__), 'Api_key.txt')
  with open(api_key_file, 'r') as f:
    api_key = f.readline().strip()
  return api_key

# Initialize the OpenAI API client
openai.api_key = read_api_key()