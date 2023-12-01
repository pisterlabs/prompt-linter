import os
import openai


openai.api_key = os.getenv("OPENAI_API_KEY")
COMPLETION_MODEL = "text-davinci-003"


prompt = """
Man Utd must win trophies, says Ten Hag ahead of League Cup final

请将上面这句话的人名提取出来，并用json的方式展示出来
"""


def get_response(prompt):
    completions = openai.Completion.create (
        engine=COMPLETION_MODEL,
        prompt=prompt,
        max_tokens=512,
        n=1,
        stop=None,
        temperature=0.0,
    )
    message = completions.choices[0].text
    return message

print(get_response(prompt))