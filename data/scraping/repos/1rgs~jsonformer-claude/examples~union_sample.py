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
            "definitions": {
                "driver": {
                    "type": "object",
                    "properties": {
                        "person_type": {"type": "string", "enum": ["DRIVER"]},
                        "first_name": {"type": "string"},
                        "last_name": {"type": "string"},
                        "driving_skill": {"type": "number"}
                    }
                },
                "passenger": {
                    "type": "object",
                    "properties": {
                        "person_type": {"type": "string", "enum": ["PASSENGER"]},
                        "first_name": {"type": "string"},
                        "last_name": {"type": "string"},
                        "likliehood_to_throwup_out_of_100": {"type": "number"}
                    }
                }
            },
            "properties": {
                "is_car_slow": {
                    "type": "boolean"
                },
                "car_model_year": {
                    "type": "number",
                    "max": 2000,
                    "min": 1940
                },
                "car_name": {
                    "type": "string",
                },
                "car_type": {
                    "type": "string",
                    "enum": ["convertible", "speedy car", "electric"]
                },
                "people_in_car": {
                    "type": "array",
                    "items": {
                        "discriminator": {
                            "propertyName": "person_type",
                            "mapping": {
                                "DRIVER": "#/definitions/driver",
                                "PASSENGER": "#/definitions/passenger"
                            }
                        }
                    }
                }
            },
        },
        prompt="Generate info about an old car with three drivers and four passengers.",
        debug=True,
    )

    print(await gen_json())

    # print gen_json.llm_request_count
    print("request count", gen_json.llm_request_count)


if __name__ == "__main__":
    asyncio.run(main())
