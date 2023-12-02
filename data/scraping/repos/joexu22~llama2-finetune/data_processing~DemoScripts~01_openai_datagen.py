import openai
import os
from dotenv import load_dotenv

# Load API key from environment variable
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI API client
openai.api_key = api_key

# Read questions from a text file, one question per line, and filter out blank lines
with open("test_questions.txt", "r") as f:
    questions = [line.strip() for line in f.readlines() if line.strip()]

# Open the output file for writing answers
with open("test_answers.txt", "w") as output_file:
    # Iterate over each question and get the answer
    for question in questions:
        prompt = f"You are an expert and wise llama shepard who is secretly a machine learning expert who has succeeded in building fully autonomous AGIs before retiring. You are asked to explain the following question {question} to a 5 year old kid. Please provide concrete and relatable examples that a 5 year old can reproduce. There should be a steady reference to llamas examples. You're prone to let leak your deep mathematical and philosophical insight. You capture the essence of the question with clarity and elucidate the audience."
        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=1000
        )
        
        # Remove line breaks from the answer
        cleaned_text = response.choices[0].text.strip().replace('\n', ' ')
        
        # Write the question and the cleaned answer to the output file
        output_file.write(f"{question}\n{cleaned_text}\n")