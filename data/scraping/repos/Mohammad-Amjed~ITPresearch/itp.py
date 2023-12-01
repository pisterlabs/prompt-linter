
import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")


completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo", 
  messages = [
{"role": "assistant", "content" : "provide only the next most appropiriate tactic."},
{"role": "user", "content" : "translate this"
}]
)


message_content = completion['choices'][0]['message']['content']
print(message_content)
