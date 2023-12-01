"""
Generate answers to questions about raising babies using GPT-4

Usage: Run the script from the repo root directory
$ poetry run python src/genai/parenting_chatbot/prodigy_eval/data/generate_gpt4_answers.py

The final output file is stored in data/answers_gpt4.jsonl file
following the format: {"question": your-question, "gpt4": gpt4-answer}

"""
import os

import dotenv
import openai
import pandas as pd

from genai import MessageTemplate
from genai.eyfs import TextGenerator


DIR = "src/genai/parenting_chatbot/prodigy_eval/data/"
SYSTEM_PROMPT = DIR + "system.json"
QUESTIONS = DIR + "questions.jsonl"
OUTPUT_FILE = DIR + "answers_gpt4.jsonl"

dotenv.load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

if __name__ == "__main__":
    # Load the system prompt
    system_prompt = MessageTemplate.load(SYSTEM_PROMPT)
    # Load the questions
    questions = pd.read_json(path_or_buf=QUESTIONS, lines=True).question.to_list()
    # Generate answers
    responses = []
    for question in questions:
        prompt = MessageTemplate(role="user", content=question)
        response = TextGenerator.generate(
            model="gpt-4",
            temperature=0.6,
            messages=[system_prompt.to_prompt(), prompt.to_prompt()],
            message_kwargs=None,
            stream=False,
        )
        responses.append(response)
    # Extract only texts from each response
    answer_text = [response["choices"][0]["message"]["content"] for response in responses]
    # Write the answers to jsonl file
    pd.DataFrame({"question": questions, "gpt4": answer_text}).to_json(
        path_or_buf=OUTPUT_FILE, orient="records", lines=True
    )
