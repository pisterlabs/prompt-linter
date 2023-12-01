import ai21
import httpx
import openai
from anthropic import AsyncAnthropic, HUMAN_PROMPT, AI_PROMPT

update_scene = {
    "name": "update_scene",
    "description": "Update the scene based on the given changes",
    "parameters": {
        "type": "object",
        "properties": {
            "narrative": {
                "type": "string",
                "description": "The next chunk of narration for the players",
            },
            "image-prompt": {
                "type": "string",
                "description": "A prompt for an AI image generator describing the "
                               "background image for the updated scene",
            },
        },
        "required": ["narrative", "image-prompt"],
    },
}


def ai21_complete(api_key: str, prompt: str):
    ai21.api_key = api_key
    response = ai21.Completion.execute(
        model="j2-ultra",
        prompt=prompt,
        maxTokens=200,
        temperature=0.4,
    )
    return response


async def ai21_custom_complete(api_key: str, prompt: str):
    model_type = "j2-ultra"
    model_name = "abcdef"
    url = f"https://api.ai21.com/studio/v1/{model_type}/{model_name}/complete"

    payload = {
        "prompt": prompt,
        "temperature": 0.7,
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers)

    return response.text


async def get_narrative_update(api_key: str, prompt: str):
    openai.api_key = api_key

    completion = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "You are a Game Master leading a group of players on a role "
                        "playing game"},
            {"role": "user", "content": prompt},
        ],
        functions=[
            update_scene
        ],
    )

    return completion.choices[0].message


async def claude_complete(api_key: str, prompt: str):
    anthropic = AsyncAnthropic(api_key=api_key)
    completion = await anthropic.completions.create(
        model="claude-1-100k",
        max_tokens_to_sample=300,
        prompt=f"{HUMAN_PROMPT} {prompt}{AI_PROMPT}",
    )
    return completion.completion
