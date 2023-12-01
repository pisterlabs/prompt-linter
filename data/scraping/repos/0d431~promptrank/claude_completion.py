import os
import anthropic
from tenacity import retry, stop_after_attempt, wait_random_exponential


@retry(wait=wait_random_exponential(min=0.5, max=20), stop=stop_after_attempt(6))
def get_claude_completion(
    system: str = "",
    prompt: str = "",
    model="claude-instant-1",
    temperature=0.0,
    max_tokens=50,
    stop=anthropic.HUMAN_PROMPT,
) -> str:
    """Run a prompt completion with Claude, retrying with backoff in failure case."""
    try:
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        response = client.completions.create(
            prompt=f"{system}{anthropic.HUMAN_PROMPT}{prompt}{anthropic.AI_PROMPT}",
            stop_sequences=[stop],
            model=model,
            max_tokens_to_sample=max_tokens,
            temperature=temperature,
        )

        return response.completion
    except Exception as ex:
        raise ex
