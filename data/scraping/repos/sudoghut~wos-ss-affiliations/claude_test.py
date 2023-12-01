from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

with open('test.txt', 'r') as file:
    test = file.read()

with open('api_key.txt', 'r') as file:
    api_key_str = file.read()


anthropic = Anthropic(
    api_key=api_key_str,
)

completion = anthropic.completions.create(
    # model="claude-2",
    model="claude-1",
    max_tokens_to_sample=300,
    # prompt=f"{HUMAN_PROMPT} how does a court case get to the Supreme Court? {AI_PROMPT}",
    prompt=f"{HUMAN_PROMPT} {test} {AI_PROMPT}",
)

print(completion.completion)
print(type(completion.completion))

