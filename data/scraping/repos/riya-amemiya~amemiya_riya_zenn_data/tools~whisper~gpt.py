import guidance
from dotenv import load_dotenv

load_dotenv()


with open("result.txt") as f:
    long_text = f.read()

guidance.llm = guidance.llms.OpenAI("gpt-3.5-turbo-16k-0613")

prompt = guidance(
    """{{#system~}}
あなたは、記事を要約して読みやすくするライターです。
{{~/system}}
{{#user~}}
{{long_text}}
{{~/user}}
{{#assistant~}}
{{gen 'summary' temperature=0.0 max_tokens=8000}}
{{~/assistant}}
"""
)
print(prompt(long_text=long_text))
with open("summary.txt", "w") as f:
    f.write(prompt(long_text=long_text)["summary"])
