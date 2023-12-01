import os
import anthropic

MAX_TOKENS = 300
# From https://console.anthropic.com/docs/api/reference#parameters
MODELS: dict[str,str] = {
    "claude-instant-v1.1-100k": (
        100_000,
        "An enhanced version of claude-instant-v1.1 with a 100,000 token"
        "context window that retains its lightning fast 40 word/sec "
        "performance."
    ),
    "claude-instant-v1-100k": (
        100_000,
        "An enhanced version of claude-instant-v1 with a 100,000 token context "
        "window that retains its performance. Well-suited for high throughput "
        "use cases needing both speed and additional context, allowing deeper "
        "understanding from extended conversations and documents."
    ),
    "claude-v1.3-100k": (
        100_000,
        "An enhanced version of claude-v1.3 with a 100,000 token (roughly "
        "75,000 word) context window."
    ),
    "claude-instant-v1.1": (
        8_000,
        "Our latest version of claude-instant-v1. It is better than "
        "claude-instant-v1.0 at a wide variety of tasks including writing, "
        "coding, and instruction following. It performs better on academic "
        "benchmarks, including math, reading comprehension, and coding tests. "
        "It is also more robust against red-teaming inputs."
    ),
    "claude-instant-v1": (
        8_000,
        "A smaller model with far lower latency, sampling at roughly 40 "
        "words/sec! Its output quality is somewhat lower than the latest "
        "claude-v1 model, particularly for complex tasks. However, it is much "
        "less expensive and blazing fast. We believe that this model provides "
        "more than adequate performance on a range of tasks including text "
        "classification, summarization, and lightweight chat applications, as "
        "well as search result summarization."
    ),
    "claude-instant-v1.0": (
        8_000,
        "An earlier version of claude-instant-v1."
    ),
    "claude-v1-100k": (
        8_000,
        "An enhanced version of claude-v1 with a 100,000 token (roughly"
        "75,000 word) context window. Ideal for summarizing, analyzing, and "
        "querying long documents and conversations for nuanced understanding "
        "of complex topics and relationships across very long spans of text."
    ),
    "claude-v1.3": (
        8_000,
        "Compared to claude-v1.2, it's more robust against red-team inputs"
        " better at precise instruction-following, better at code, and better "
        "and non-English dialogue and writing."
    ),
    "claude-v1.2": (
        8_000,
        "An improved version of claude-v1. It is slightly improved at general "
        "helpfulness, instruction following, coding, and other tasks. It is "
        "also considerably better with non-English languages. This model also "
        "has the ability to role play (in harmless ways) more consistently, "
        "and it defaults to writing somewhat longer and more thorough "
        "responses."
    ),
    "claude-v1": (
        8_000,
        "Our largest model, ideal for a wide range of more complex tasks."
    ),
    "claude-v1.0": (
        8_000,
        "An earlier version of claude-v1."
    ),
}



def calculate_tokens(prompt: str, model_version: str) -> int | None:
    if 'claude' in model_version.lower():
        return anthropic.count_tokens(prompt)
    else:
        raise Exception('UNKOWN MODEL - Cannot calculate tokens')

def ask_claude(
        prompt: str,
        max_tokens: int = MAX_TOKENS,
        model_version: str = 'claude-v1-100k',
        api_key: str | None = None,
        **anthropic_client_kwargs,
    ) -> dict[str, str]:
    '''Use Claude via API (https://console.anthropic.com/docs/api)'''
    if api_key is None:
        api_key = os.environ['ANTHROPIC_API_KEY']
    client = anthropic.Client(api_key=api_key)
    resp = client.completion(
        prompt=prompt,
        stop_sequences=[anthropic.HUMAN_PROMPT],
        model=model_version,
        max_tokens_to_sample=max_tokens,
        **anthropic_client_kwargs,
    )
    return resp

def attempt_claude_fix_json(
    problematic_json: str,
    max_tokens: int = MAX_TOKENS,
    prompt_override: str | None = None,
    **claude_kwargs,
) -> str:
    if prompt_override:
        prompt = prompt_override
    else:
        prompt = (
            f'{anthropic.HUMAN_PROMPT} '
            f'Fix the following text so JSON is properly formatted. '
            'Make sure you are careful fixing the proper JSON format '
            '(including commas, quotes, and brackets).\n'
            f'{problematic_json}\n'
            f'{anthropic.AI_PROMPT}'
        )
    # Let the kwargs override the max_tokens given explicitly or by default
    claude_kwargs['max_tokens'] = claude_kwargs.get('max_tokens', max_tokens) 
    r = ask_claude(
        prompt=prompt,
        **claude_kwargs,
    )
    return r['completion']