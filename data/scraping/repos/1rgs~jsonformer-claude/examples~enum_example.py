import asyncio
import anthropic
import os

from jsonformer_claude.main import JsonformerClaude

api_key = os.environ["ANTHROPIC"]

client = anthropic.Client(api_key)
GENERATION_MARKER = "|GENERATION|"


async def main():
    gen_json = JsonformerClaude(
        anthropic_client=client,
        max_tokens_to_sample=500,
        model="claude-instant-v1",
        json_schema={
            "properties": {
                "is_car_slow": {
                    "type": "boolean"
                },
                "car_model_year": {
                    "type": "number",
                    "max": 1950,
                    "min": 1940
                },
                "car_name": {
                    "type": "string",
                },
                "car_type": {
                    "type": "string",
                    "enum": ["convertible", "speedy car", "electric"]
                },
            },
        },
        prompt="Generate info about an old car",
        debug=True,
    )

    print(await gen_json())

    # print gen_json.llm_request_count
    print("request count", gen_json.llm_request_count)


if __name__ == "__main__":
    asyncio.run(main())
