import os
from anthropic import AsyncAnthropic, HUMAN_PROMPT, AI_PROMPT
from dotenv import load_dotenv

load_dotenv()
anthropic = AsyncAnthropic(
    api_key=os.getenv("ANTHROPIC_KEY"),
)


async def claude(messages):
    # messages are a dict {} with content and roler.
    prompt = ""
    for message in messages:
        if message["role"] == "system":
            prompt += f"{HUMAN_PROMPT} The name in brackets after \"Human \" is the username of the person sending the message\n{message['content']}"
        elif message["role"] == "assistant":
            prompt += f"{AI_PROMPT} {message['content']}"
        elif message["role"] == "user":
            prompt += f"\n\nHuman ({message['name']}): {message['content']}"
        elif message["role"] == "function":
            ...
    prompt += AI_PROMPT
    completion = await anthropic.completions.create(
        stop_sequences=["\n\nHuman (", "\n\nSYSTEM: "],
        model="claude-2",
        max_tokens_to_sample=512,
        prompt=prompt,
    )
    return {
        "name": "send_message",
        "arguments": {"message": completion.completion},
    }  # a dummy function call is created.
