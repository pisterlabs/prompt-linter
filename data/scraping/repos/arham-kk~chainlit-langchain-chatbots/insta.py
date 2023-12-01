import chainlit as cl
import openai, os

openai.api_key = "OPENAI_API_KEY"

prompt = """Generate 10 instgram caption ideas {input}:```"""

model_name = "text-davinci-003"

settings = {
    "temperature": 0.7,
    "max_tokens": 500,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "stop": ["```"]
}

@cl.on_message
async def main(message: str):
    fromatted_prompt = prompt.format(input=message)

    msg = cl.Message(
        content="",
        prompt=fromatted_prompt,
        llm_settings=cl.LLMSettings(model_name=model_name, **settings),
    )

    async for stream_resp in await openai.Completion.acreate(
        model=model_name, prompt=fromatted_prompt, stream=True, **settings
    ):
        token = stream_resp.get("choices")[0].get("text")
        await msg.stream_token(token)

    await msg.send()
