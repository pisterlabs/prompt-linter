"""
"""
import openai
from core.types import OpenAIResponse
from core.types import Claim, ArbitrationResult


openai.api_key = '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'  # cost Adam money!

_BASE_PROMPT: str = "{name_1}: {claim_1}\n\n{name_2}: {claim_2}\n\n1. Who is right, {name_1} or {name_2}?\n2. What is the reason for that person to be right?\n\n1. Right:"


async def query_openai_for_arbitration(claim_1: Claim, claim_2: Claim) -> ArbitrationResult:
    """Calling OpenAI API with a prompt and extract the relevant info

    :param claim_1:
    :param claim_2:

    :return: `ArbitrationResult`
    """
    response: OpenAIResponse = openai.Completion.create(
        model="text-davinci-002",
        prompt=_BASE_PROMPT.format(
            name_1=claim_1["person"],
            claim_1=claim_1["claim"],
            name_2=claim_2["person"],
            claim_2=claim_2["claim"]
        ),
        temperature=0.7,
        max_tokens=2735,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    print(f'OpenAI response is: {response}')
    raw_text_response = response['choices'][0]['text'] or ''
    right, reason, = raw_text_response.split('Reason:', maxsplit=1)
    return ArbitrationResult(right=right, reason=reason)
