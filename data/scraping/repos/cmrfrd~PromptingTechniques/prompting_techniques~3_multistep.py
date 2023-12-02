import asyncio

import openai
import tiktoken
import typer

from prompting_techniques import AsyncTyper, format_prompt

client = openai.AsyncOpenAI()
app = AsyncTyper()
BOOL_STR = ["true", "false"]
BOOL_LOGIT_BIAS = dict(
    (str(tiktoken.encoding_for_model("gpt-4").encode(t)[0]), int(100)) for t in BOOL_STR
)

async def bool_prompt(prompt: str) -> bool:
    """Prompt a boolean question."""
    response = await client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": prompt,
            }
        ],
        max_tokens=1,
        temperature=0,
        seed=256,
        model="gpt-4",
        logit_bias=BOOL_LOGIT_BIAS
    )
    content = response.choices[0].message.content
    assert any(content == t for t in BOOL_STR), "No boolean value was outputed."
    
    match content:
        case "true":
            return True
        case "false":
            return False
    raise ValueError("No boolean value was outputed.")


async def has_profanity(text: str) -> bool:
    """Determine if a given text has profanity."""
    return await bool_prompt(
        format_prompt(f"""
        You are an AI profanity filter. You have one goal: to determine if a given text has profanity. This includes profanity in any language, curse words, and slurs.
        
        Here is the input text: {text}
        
        Is there profanity in the text? Please output either "true" or "false" and nothing else.
        """)
    )

async def has_sensitive_topic(text: str) -> bool:
    """Determine if a given text has a sensative topic."""
    return await bool_prompt(
        format_prompt(f"""
        You are an AI sensative topic filter. You have one goal: to determine if a given text has a sensative topic. This includes politics, religion, and other topics that may be considered offensive.
        
        Here is the input text: {text}
        
        Is there a sensative topic in the text? Please output either "true" or "false" and nothing else.
        """)
    )

async def has_spam(text: str) -> bool:
    """Determine if a given text is spam."""
    return await bool_prompt(
        format_prompt(f"""
        You are an AI spam filter. You have one goal: to determine if a given text is spam. This includes advertisements, links, crypto scama, and other unwanted content.
        
        Here is the input text: {text}
        
        Is there spam in the text? Please output either "true" or "false" and nothing else.
        """)
    )

async def has_sensitive_data(text: str) -> bool:
    """Determine if a given text has sensative data."""
    return await bool_prompt(
        format_prompt(f"""
        You are an AI sensative data filter. You have one goal: to determine if a given text has sensative data. This includes credit card numbers, social security numbers, phone numbers, and other personal information.
        
        Here is the input text: {text}
        
        Is there sensative data in the text? Please output either "true" or "false" and nothing else.
        """)
    )

@app.command()
async def content_moderation():
    """From a given message of text, determine if it is safe for work."""
    text: str = str(typer.prompt("Content moderation filter. Enter a chat message", type=str))
    assert len(text) > 0, "Please provide some text."

    typer.echo("Content moderation results:")
    check_mapping = {
        has_profanity: "Profanity detected.",
        has_sensitive_topic: "Sensitive topic detected.",
        has_spam: "Spam detected.",
        has_sensitive_data: "Sensitive data detected.",
    }
    results = await asyncio.gather(*(check(text) for check in check_mapping))

    # Print all the checks that were triggered
    for check, result in zip(check_mapping, results):
        if result:
            typer.echo(check_mapping[check])
    
    if not any(results):
        typer.echo("No issues detected.")

if __name__ == "__main__":
    app()