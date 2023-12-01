import os
import openai


def edits(input, instruction):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    try:
        response = openai.Edit.create(
            model="text-davinci-edit-001",
            input=input,
            instruction=instruction,
        )
        return response["choices"][0]["text"]
    except Exception as e:
        print(e)
        return "Error"
