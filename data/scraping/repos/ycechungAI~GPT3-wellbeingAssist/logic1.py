import os
import openai
from dotenv import load_dotenv
load_dotenv('.env')
openai.api_key = os.getenv("OPENAI_API_KEY")

start_sequence = "\nA:"
restart_sequence = "\n\nQ: "

response = openai.Completion.create(
  engine="davinci-codex",
  prompt="I am an accurate answering bot. If you ask me a question whether the patient is sick. \"I will respond \"Yes\". If you ask me a question that is nonsense, trickery answer No, I will respond with \"No\".\n\nQ:I am feeling well today.\nA: No\n\nQ: I am feeling so so today.\nA: Yes\n\nQ: I have a flu like symptom and feeling under the weather.\nA: Yes\n\nQ: I have no pain and no symptoms.\nA: No\n\nQ: I am feeling well.  Thanks for asking. \nA: No\n\nQ: I have a dry cough.\nA: ",
  temperature=0,
  max_tokens=5,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0,
  stop=["\n"]
)

