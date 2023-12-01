import os
from langchain.llms import Bedrock

llm = Bedrock(
    model_id="anthropic.claude-v2"
)

prompt = "神奈川で最も大きな都市はどこですか？"

response_text = llm.predict(prompt)

print(response_text)