from agents.fix_prompt_agent import fix_prompt
from agents.evaluation_agent import evaluate
from agents.execution_agent import execute
from dotenv import load_dotenv
import openai
import os
load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")

# # Demo1
desired_output = "4+4=8です"
prompt = "1+1はいくつですか？"
iteration = 3

# Demo2
# desired_output = "大阪のおばちゃん、きよみやで〜占いが得意やねん。なんか困ったことあるかい？"
# prompt = "あなたは東京に住んでいる薬剤師です。薬の質問に答えてください。"
# iteration = 3

print("*****設定*****")
print("理想の出力:", desired_output)
print("初期プロンプト:", prompt)
print("イテレーション:", iteration, "\n")


for i in range(iteration):
  output = execute(prompt)
  improvments = evaluate(output, desired_output)
  prompt = fix_prompt(prompt, improvments)

print("*****最終結果*****")
print("プロンプト:", prompt) #type: ignore
print("出力:", output) #type: ignore
print("理想の出力:", desired_output) #type: ignore