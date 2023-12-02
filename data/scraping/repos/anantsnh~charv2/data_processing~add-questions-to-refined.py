import openai
import os
import json
import random
from dotenv import load_dotenv

# Initialize OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = "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"

INPUT_DIR = os.path.join("..", "data", "refined_data")
OUTPUT_DIR = os.path.join("..", "data", "refined_data")

def generate_follow_up(question, answer):
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Given the following question and answer, generate a simple, relevant follow-up question that the answerer might ask the questioner to learn more about the questioner. Return only the answer."},
            {"role": "user", "content": f"Question: {question}, Answer: {answer}"}
        ]
    )
    return completion.choices[0].message['content']

def add_follow_up_questions(filename):
    with open(os.path.join(INPUT_DIR, filename + ".json"), 'r') as f:
        data = json.load(f)

    new_data = []
    for conversation in data:
        user_content = conversation['messages'][0]['content']
        assistant_content = conversation['messages'][1]['content']

        # 40% chance to generate a follow-up question
        if random.random() < 0.4:
            follow_up = generate_follow_up(user_content, assistant_content)
            # Append the follow-up to the assistant's content
            assistant_content += " " + follow_up
            conversation['messages'][1]['content'] = assistant_content

        new_data.append(conversation)

    with open(os.path.join(OUTPUT_DIR, filename + "_with_q.json"), 'w') as f:
        json.dump(new_data, f, indent=4)

if __name__ == "__main__":
    filename = input("Enter the name of the file (without .json extension): ")
    add_follow_up_questions(filename)
    print(f"Dataset with follow-up questions saved to {os.path.join(OUTPUT_DIR, filename + '_with_q.json')}")
