import os

api_key = os.getenv("OPENAI_API_KEY")

# Now, you can use the `api_key` variable to authenticate with the OpenAI API.

import os
import openai

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")

chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello world"}])