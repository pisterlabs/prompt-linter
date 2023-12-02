# ペットの名前を5つ考えて表示する
import openai, os

# APIキーを環境変数から設定 --- (*1)
openai.api_key = os.environ["OPENAI_API_KEY"]

# ChatGPTのAPIを呼び出す --- (*2)
def call_chatgpt(prompt, debug=False):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{'role': 'user', 'content': prompt}]
    )
    # ChatGPTからの応答内容を全部表示 --- (*3)
    if debug: print(response)
    # 応答からChatGPTの返答を取り出す --- (*4)
    content = response.choices[0]['message']['content']
    return content

# ペットの名前を生成して表示 --- (*5)
pet_names = call_chatgpt('ペットの名前を5つ考えて', debug=False)
print(pet_names)
