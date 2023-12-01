import asyncio
import anthropic

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
            "type": "object",
            "properties": {
                "car": {
                    "type": "object",
                    "properties": {
                        "make": {"type": "string"},
                        "model": {"type": "string"},
                        "year": {"type": "number"},
                        "VIN": {"type": "string"},
                        "colors": {"type": "array", "items": {"type": "string"}},
                        "features": {
                            "type": "object",
                            "properties": {
                                "audio": {
                                    "type": "object",
                                    "properties": {
                                        "brand": {"type": "string"},
                                        "speakers": {"type": "number"},
                                        "hasBluetooth": {"type": "boolean"},
                                    },
                                },
                                "safety": {
                                    "type": "object",
                                    "properties": {
                                        "airbags": {"type": "number"},
                                        "parkingSensors": {"type": "boolean"},
                                        "laneAssist": {"type": "boolean"},
                                        "adaptiveCruiseControl": {"type": "boolean"},
                                        "blindSpotMonitoring": {"type": "boolean"},
                                    },
                                },
                                "performance": {
                                    "type": "object",
                                    "properties": {
                                        "engine": {"type": "string"},
                                        "horsepower": {"type": "number"},
                                        "topSpeed": {"type": "number"},
                                        "zeroToSixty": {"type": "number"},
                                    },
                                },
                            },
                        },
                        "maintenanceRecords": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "date": {"type": "string", "format": "date"},
                                    "servicePerformed": {"type": "string"},
                                    "partsReplaced": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                            },
                        },
                    },
                },
                "owner": {
                    "type": "object",
                    "properties": {
                        "firstName": {"type": "string"},
                        "lastName": {"type": "string"},
                        "age": {"type": "number"},
                        "licenseNumber": {"type": "string"},
                        "address": {
                            "type": "object",
                            "properties": {
                                "street": {"type": "string"},
                                "city": {"type": "string"},
                                "state": {"type": "string"},
                                "zip": {"type": "string"},
                            },
                        },
                        "contactInfo": {
                            "type": "object",
                            "properties": {
                                "email": {"type": "string", "format": "email"},
                                "phoneNumber": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
        prompt="Generate info about a car",
        debug=True,
    )

    print(await gen_json())

    # print gen_json.llm_request_count
    print("request count", gen_json.llm_request_count)


if __name__ == "__main__":
    asyncio.run(main())
