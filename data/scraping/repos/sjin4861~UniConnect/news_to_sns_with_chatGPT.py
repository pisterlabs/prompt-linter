import openai
import csv
import os
from dotenv import load_dotenv

load_dotenv()

# API 키 로드
api_key = os.getenv("OPENAI_API_KEY")

# API 키가 올바르게 로드되었는지 확인
if api_key is None:
    raise ValueError("API 키가 .env 파일에서 로드되지 않았습니다.")

# 이제 api_key를 사용하여 OpenAI API에 액세스할 수 있습니다.

# CSV 파일에서 제목과 내용을 읽어와 리스트로 저장
def read_csv(file_path):
    data = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append((row['제목'], row['내용']))
    return data

# GPT 모델을 사용하여 글을 재작성
def generate_text(messages):
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=1000,  # 생성할 최대 토큰 수
        temperature=0.7
    )
    return response.choices[0].message['content']

# CSV 파일에서 데이터 읽어오기
csv_file = '/Users/jun/Desktop/my_project/UniConnect/Data/crawled_news.csv'  # 실제 파일 경로로 바꿔주세요
data = read_csv(csv_file)

conversation = [
    {"role": "system", "content": "넌 세계의 각 대학과 입시를 준비하는 사람을 잇는 SNS 플랫폼의 메신저야. 제목과 내용을 입력하면 사용자들에게 지적 호기심을 불러일으킬 수 있는 글을 재 작성해줘."},
    {"role": "user", "content": f"제목: {data[3][0]}\n내용: {data[3][1]}"},
    #{"role": "assistant", "content": "Sure! We have several exciting research projects, including..."}
]

new_text = generate_text(conversation)
    
# 생성된 글을 원하는 곳에 저장 또는 활용
# 여기서는 콘솔에 출력하겠습니다.
print(new_text)
