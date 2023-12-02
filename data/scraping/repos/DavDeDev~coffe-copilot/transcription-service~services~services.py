import cohere
import os
from dotenv import load_dotenv

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.Client(COHERE_API_KEY)

def generate_conversation_prompts(background, conversation_summary):
    prompt = background + "\n" + conversation_summary

    response = co.chat(message=prompt, connectors=[{"id": "web-search"}], max_tokens=1000, prompt_truncation="AUTO")

    return response.text


def generate_conversation_summary(conversation):
    response = co.summarize(text=conversation, length='medium',
                            format='paragraph',
                            model='summarize-xlarge',
                            additional_command='Separate into Bullet Points',
                            temperature=0.3,
                            )

    return response.summary