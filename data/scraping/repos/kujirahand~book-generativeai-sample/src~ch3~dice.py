# ChatGPTをサイコロとして使う
import openai, os

# APIキーを環境変数から設定 --- (*1)
openai.api_key = os.environ['OPENAI_API_KEY']

# ChatGPTのAPI(Completion)を呼び出す --- (*2)
def completion(prompt, debug=False):
    response = openai.Completion.create(
        model='text-davinci-003',
        prompt=prompt,
        temperature=1.0 # ランダム性 --- (*3)
    )
    # ChatGPTからの応答内容を全部表示
    if debug: print(response)
    # 応答からChatGPTの返答を取り出す --- (*4)
    content = response['choices'][0]['text'].strip()
    return content

if __name__ == '__main__':
    # サイコロになりきってもらう --- (*5)
    result = completion(
        prompt='''
        あなたはサイコロです。
        ランダムに1以上6以下の数字を1つ選んでください。
        ''',
        debug=False)
    print(result)
