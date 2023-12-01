import openai

openai.api_key = "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" 


def Ask_ChatGPT(message):
    
    # 応答設定
    completion = openai.ChatCompletion.create(
                 model    = "gpt-3.5-turbo",    
                 messages = [{
                     "role":"system",
                     "content":'あなたは優秀なアシスタントです。'},
                    {
                     "role":"user",      
                     "content":message,   
                            }],
    
                 max_tokens  = 1024,            
                 n           = 1,                
                 stop        = None,            
                 temperature = 0.5,              
    )
    
    # 応答
    response = completion.choices[0].message["content"]
    
    # 応答内容出力
    return response


# 目標の取得
goal = input("あなたの達成したい目標は何ですか？: ")

# 期限の取得
deadline = input("その目標の達成期限はいつですか？: ")

# GPT-3に1週間のスケジュールを生成させる
prompt = f"私は{goal}を{deadline}までに達成したい。今日からの1週間のスケジュールをjson形式で提案してください。"
massage = Ask_ChatGPT(prompt)
# print(response)
print(massage)