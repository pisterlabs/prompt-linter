import os
import openai
from dotenv import load_dotenv

load_dotenv()

# Load your API key from an environment variable or secret management service
openai.api_key = os.environ.get("OPENAI_API_KEY")

response = openai.Completion.create(model="text-davinci-003", prompt="Generate a script about the life of Will Smith",
                                    temperature=0.1, n=1, max_tokens=2048, top_p=1, frequency_penalty=0, presence_penalty=0)

print(response)

response.get("text")