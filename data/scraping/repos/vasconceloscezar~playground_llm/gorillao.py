import os
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = "EMPTY"
# openai.api_key = os.getenv("OPENAI_API_KEY_2")
openai.api_base = "http://34.132.127.197:8000/v1"
model = "gorilla-7b-hf-v0"

PROMPT = "I want to generate a cat image, using a stable diffusion model."
completion = openai.ChatCompletion.create(
    model=model,
    temperature=0.0,
    messages=[{"role": "user", "content": PROMPT}],
)

response = completion.choices[0].message.content
print(f"Resposta  do Gorilao: {response}")

# Extract code segments and save to a Python file
code_blocks = []
is_code_block = False

for line in response.split("\n"):
    if line.startswith("<<<code>>>"):
        is_code_block = True
    elif line.startswith("<<<"):
        is_code_block = False
    elif is_code_block:
        code_blocks.append(line.strip())

if code_blocks:
    with open("generated_code.py", "w") as file:
        file.write("\n".join(code_blocks))
    print("Code segments saved to 'generated_code.py'")
else:
    print("No code segments found in the response.")
