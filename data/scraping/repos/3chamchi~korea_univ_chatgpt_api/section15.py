# Section15
# 상품 설명 생성기

# 1. API 요청 준비
import openai  # or request

openai.api_key = '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'  # API Key

question1 = input('1. 제품명 : ')  # API 요청에 보낼 내용 입력
question2 = input('2. 주요 이용 고객 : ')
question3 = input('3. 제품 특징 : ')
question4 = input('4. 경쟁 제품 대비 차별점(좋은점) : ')

# list [], dict {}, str(문자열) '' or ""
messages = [  # messages list 자료형
    {  # messages[0]
        'role': 'system',  # system str() 문자열 자료형
        'content': '상품 설명을 만드는 역할이야. user가 입력한 내용을 참고해서 상품 설명을 만들어줘. 2가지 버전을 제안해줘.',
    },
    {  # messages[1]
        'role': 'user',  # user 프로그램 사용자가 입력 의미
        'content': f'1. 제품명 : {question1}, 2. 주요 이용 고객 : {question2}, 3. 제품 특징 : {question3}, 4. 경쟁 제품 대비 차별점(좋은점) : {question4}',
    }  # '1. 제품명 : ' + question1 + ', 2. 주요 이용 고객 : ' + question2 + '3. 제품 특징 : ' + question3
]
# list 인덱스로 접근, dict 키값으로 접근
print(messages)
print(messages[0]['content'])
print(type(messages))

# 2. API 요청
response = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=messages)

# 3. API 응답 확인
print(response.to_dict())
print(response.choices[0].message['content'])
