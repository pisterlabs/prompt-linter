import os
import openai
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

# Set the OpenAI API key from the environment variables
openai.api_key = os.getenv("OPENAIKEY")


def generate_answer(prompt: str,max_tokens) -> str:
    """
    Generate an answer based on the given prompt using ChatGPT.

    :param prompt: The prompt to generate the answer from.
    :return: The generated answer as a string.
    """
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.9,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0.1,
        presence_penalty=0.6,
        # stop=[" Human:", " AI:"],
    )

    answer = response["choices"][0]["text"]
    print(response)
    return answer

# Example usage
# question = "What is the capital of France?"
# answer = generate_answer(prompt)
# print(answer)
