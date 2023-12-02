import os

from transformers import pipeline
import openai
from dotenv import load_dotenv

load_dotenv()
# input_file = open("prompts/chat-with-skit.txt", "r")


def get_response(input_prompt):
    """This function is used to generate the response from the
    openAI chatGPT model API

    :param input_prompt: The input prompt for the model
    :type input_prompt: str
    :return: str -- The generated response from the model
    :raises: None
    """
    checkpoint = "MBZUAI/LaMini-Neo-125M"
    model = pipeline(
        "text-generatin",
        model=checkpoint,
    )
    generated_text = model(
        input_prompt,
        max_length=512,
        do_sample=True,
    )[
        0
    ]["generated_text"]
    return generated_text


system = """Transcript of a dialog, where the Customer interacts with a Marketing Employee named AI, who works at Skit.
            AI is supposed to convince the Customer to renew the subscription to a service. Start the conversation by convincing the Customer to renew the subscription.
            """


def get_response_gpt(input_prompt):
    """This function is used to generate the response from the
    openAI chatGPT model API
    :param input_prompt: The input prompt for the model
    :type input_prompt: str
    :return: str -- The generated response from the model
    :raises: None
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0301",
        messages=[
            {
                "role": "system",
                "content": system,
            },
            {
                "role": "user",
                "content": input_prompt,
            },
        ],
        stop=["\n", "AI:", "Customer:"],
        max_tokens=50,
    )

    return completion.choices[0].message


test = """Customer: Hello.
AI: Hello. I am AI, talking to you on behalf of Skit.
Customer: Ohh Okay, Why did you call?
AI: 
"""
print(get_response_gpt(test))
