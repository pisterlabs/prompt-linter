import weaviate
import json
from api_keys import OPENAI_API_KEY

auth_config = weaviate.AuthApiKey(api_key="readonly-demo")

client = weaviate.Client(
        url="https://edu-demo.weaviate.network",
        auth_client_secret=auth_config,
        additional_headers={
            "X-OpenAI-Api-Key" : OPENAI_API_KEY
            }
        )
ask = {
  "question": "When was the Indian Olympics?",
  "properties": ["wiki_summary"]
}

res = (
  client.query
  .get("WikiCity", [
      "city_name",
      "_additional {answer {hasAnswer property result} }"
  ])
  .with_ask(ask)
  .with_limit(1)
  .do()
)

with open("indian_olympics.txt", "w") as file:
    file.write(json.dumps(res, indent=2))
