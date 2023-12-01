# 桃太郎が犬を仲間にできるかどうかを点数判定するゲーム
import openai, json, os
# APIキーを環境変数から設定
openai.api_key = os.getenv('OPENAI_API_KEY')
# ---------------------------------------------------------
# ゲームで使うプロンプトのテンプレートを指定 --- (*1)
template = '''
次の題の文章について、論理的かどうか、ユニークかどうかを0から100で採点してください。

### 題
- 桃太郎が鬼退治に行く仲間を探す

### 応答の例
{"論理":80, "ユニーク": 30, "論評": "論理的だが、ありふれた内容で、心が動かない"}
{"論理":50, "ユニーク": 90, "論評": "論理的ではないが、ユニークで面白い"}

### 文章
"""__MSG__"""
'''
# ---------------------------------------------------------
# ChatGPTのAPIを呼び出す --- (*2)
def chat_completion(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages)
    # 応答からChatGPTの返答を取り出して返す
    return response.choices[0]['message']['content']

# ---------------------------------------------------------
# 繰り返し説得を試みる --- (*3)
point = 0
print('犬を見つけました。犬を仲間にしたいので説得しましょう！')
while True:
    msg = input('>>> ') # ユーザーからの入力を得る
    # messagesオブジェクトを組み立てる --- (*4)
    prompt = template.replace('__MSG__', msg.replace('"', ''))
    messages = [
        {'role': 'system', 'content': 'JSONで応答してください。'},
        {'role': 'user', 'content': prompt}
    ]
    # ChatGPTによる応答を取得 --- (*5)
    s = chat_completion(messages)
    try:
        logic, unique, comment = 0, 0, '?'
        res = json.loads(s)
        if '論理' in res: logic = res['論理']
        if 'ユニーク' in res: unique = res['ユニーク']
        if '論評' in res: comment = res['論評']
        point += logic + unique
    except:
        print('[エラー] JSONの解析に失敗しました。', s)
        continue
    # ChatGPTの応答を表示 --- (*6)
    print(f'論理: {logic}点, ユニーク: {unique}点 → {comment}')
    print(f'--- 合計得点: {point} ---')
    if point >= 300:
        print('犬が仲間になってくれました！')
        print('ゲームクリア！')
        break # ゲームを終了する
    else:
        print('引き続き説得しましょう。')
