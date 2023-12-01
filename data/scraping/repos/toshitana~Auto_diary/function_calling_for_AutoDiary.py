import openai
import json

# Step 1, send model the user query and what functions it has access to
def make_three_questions():
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[{"role": "user", "content": "あなたは優秀な日記生成アシスタントAIです。あなたは日記作成のネタとして、3つの質問を行ってください。"}],
        functions=[
            {
                "name": "show_three_questions",
                "description": "日記作成のための3つの質問を表示します",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question1": {
                            "type": "string",
                            "description": "1つ目の質問"
                        },
                        "question2": {
                            "type": "string",
                            "description": "2つ目の質問"
                        },
                        "question3": {
                            "type": "string",
                            "description": "3つ目の質問"
                        }
                    }
                }
            }
        ],
        function_call="auto",
    )
    message = response["choices"][0]["message"]
    result = json.loads(message["function_call"]["arguments"])
    return result

def generate_diary(question1,question2,question3,answer1,answer2,answer3):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[{"role": "user", "content": """
        あなたは優秀な日記生成アシスタントAIです。
        あなたは下記の質問と答えを元に日記を生成してください
        {question1}:{answer1}
        {question2}:{answer2}
        {question3}:{answer3}
        """.format(question1=question1,answer1=answer1,question2=question2,answer2=answer2,question3=question3,answer3=answer3)}],
        functions=[
            {
                "name": "generate_diary",
                "description": "日記を作成します",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "diary": {
                            "type": "string",
                            "description": "生成された日記"
                        }
                    }
                }
            }
        ],
        function_call="auto",
    )
    message = response["choices"][0]["message"]
    result = json.loads(message["function_call"]["arguments"])
    return result


questions = make_three_questions()
print(questions)
print(type(questions))
question1 = questions["question1"]
question2 = questions["question2"]
question3 = questions["question3"]

answer1 = input()
answer2 = input()
answer3 = input()

print(generate_diary(question1,question2,question3,answer1,answer2,answer3))