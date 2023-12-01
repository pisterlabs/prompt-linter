import os
import openai
from dotenv import load_dotenv
load_dotenv()

openai.api_type = os.getenv("OPENAI_API_TYPE")
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_version = os.getenv("OPENAI_API_VERSION")

OPENAI_API_DEPLOYMENT_ID = os.getenv("OPENAI_API_DEPLOYMENT_ID")


def predict(data: dict):
    response = openai.Completion.create(
        engine="vchar-curie",
        prompt=f"""You are a helpful assistant that helps users of Unix systems (linux/mac and the like) with their terminal execution issues. The user will provide input/error info and you're supposed to reply with suggestions to help them fix the problem. It could be alternate commands, missing flags, and the like. If you're not sure, suggest the user to read more about a particular term. Please suggest a correct command if you have one and then a brief explanation.n\nDetails: MaxTokens = 150+{"; ".join(list(map(lambda i: ": ".join(i), data.items())))} \n\nExplanation: """,
        temperature=0.7,
        max_tokens=250,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        best_of=1,
        stop=None)
    return response["choices"][0]["text"]
