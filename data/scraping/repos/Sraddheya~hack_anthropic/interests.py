from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

anthropic = Anthropic(api_key="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")
completion = anthropic.completions.create(
    model="claude-2",
    max_tokens_to_sample=300,
    prompt=f"{HUMAN_PROMPT} How many toes do dogs have?{AI_PROMPT}",
)
print(completion.completion)