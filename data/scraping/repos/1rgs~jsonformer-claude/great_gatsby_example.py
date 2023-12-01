import asyncio
import os
import anthropic
import requests
import json
from jsonformer_claude.main import JsonformerClaude

api_key = os.environ["ANTHROPIC"]
client = anthropic.Client(api_key)
GENERATION_MARKER = "|GENERATION|"


async def main():
    # Fetch "The Great Gatsby" text from the URL
    url = "http://gutenberg.net.au/ebooks02/0200041.txt"
    response = requests.get(url)
    text = response.text

    gen_json = JsonformerClaude(
        anthropic_client=client,
        max_tokens_to_sample=2000,
        model="claude-v1",
        json_schema={
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "characters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                        },
                    },
                },
            },
        },
        prompt=f"Generate names and descriptions for all characters, even minor ones in the following book:\n {text}",
        debug=True,
    )

    print(json.dumps(await gen_json(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
