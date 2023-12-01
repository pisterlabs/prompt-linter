import guidance
import json

# set the default language model used to execute guidance programs
guidance.llm = guidance.llms.OpenAI("text-davinci-003")

instruct = guidance("""
あなたは優秀な日記生成アシスタントAIです。あなたは日記作成のネタとして、3つの質問を行ってください。

$JSON_BLOB = {
    "question1": "質問1",
    "question2": "質問2",
    "question3": "質問3"
}

Question list json:{{gen 'result' n=1 temperature=0.8 max_tokens=256}}
""")
executed_program = instruct()

print(executed_program["result"])

data = json.loads(executed_program["result"])

print(type(data))