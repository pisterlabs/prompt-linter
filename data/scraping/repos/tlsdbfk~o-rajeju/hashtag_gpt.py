import openai
import pandas as pd

openai.api_key = "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"

data = pd.read_csv("음식점_1.csv", sep=",")

# 기존 데이터프레임 생성
df = pd.DataFrame()

for i in range(965, len(data)):
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        temperature = 0.2,
        max_tokens = 1000,
        messages = [
            {"role": "user", 
            "content": str(data["개요"][i]) + "에서 중요한 내용을 해시태그 형태로 알려줘"}
            ]
        )

    # 새로운 행 추가
    new_row = {'명칭' : str(data["명칭"][i]),
        '개요' : response['choices'][0]['message']['content']}

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    print(df)

# 첫 번째 CSV 파일 읽기
data1 = pd.read_csv('음식점_1.csv')

# 두 개의 데이터를 이어붙이기
concatenated_data = pd.concat([data1, df])

# 결과를 새로운 CSV 파일로 저장
concatenated_data.to_csv('음식점_해시태그.csv')