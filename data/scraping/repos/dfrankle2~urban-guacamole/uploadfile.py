import openai
openai.api_key = "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"
response = openai.File.create(file=open("knowledge_jsonl2.jsonl"), purpose="answers")
print(response)
