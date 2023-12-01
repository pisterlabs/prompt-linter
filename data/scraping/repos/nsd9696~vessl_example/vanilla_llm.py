from langchain.llms import OpenAI
import os

OPENAI_API_KEY=os.environ["OPENAI_API_KEY"]
llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)

prefix: str = "PROMPT"
prompts = []
for key, value in os.environ.items():
    if key[:len(prefix)] == prefix:
        prompts.append(value)

test_prompts = ["What is Haerae?", "What is special in Haerae?"]
prompts += test_prompts

for prompt in prompts:
    completion = llm(prompt)
    print(f"⏳ Prompt: {prompt} is querying...")
    print(f"✅ Resonpse: {completion}")
