# Section10 - 2
# ChatGPT API 연동 openai 패키지 사용

# 1. API 연동 준비
import openai

openai.api_key = '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'

# 2. API 요청
pre_lang = input('입력하는 언어를 입력해주세요 : ')
to_lang = input('변환하려는 언어를 입력해주세요 : ')
text = input('번역할 내용을 입력해주세요 : ')

response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    messages=[
        {
            'role': 'system',  # system, user, assistant
            'content': f'{pre_lang}를 {to_lang}로 번역해줘.',
        },
        {
            'role': 'user',
            'content': text,
        }
    ]
)

# 3. 응답 값 확인
print(response.to_dict())  # openai의 클래스인 경우
