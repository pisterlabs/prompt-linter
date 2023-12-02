from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

claude = Anthropic()
prompt = f"{HUMAN_PROMPT} Is 2 + 2 = 4? Explain your answer.{AI_PROMPT} No,"
response = claude.completions.create(
    model="claude-2.0",
    max_tokens_to_sample=1024,
    prompt=prompt,
    temperature=0.0,
)
print(prompt + response.completion)
