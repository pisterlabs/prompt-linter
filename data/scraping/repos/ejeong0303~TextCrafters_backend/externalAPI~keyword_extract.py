import openai
import os

openai.api_key = "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"

review = ("점심시간에 오기 좋은 이대인도음식 맛집이에요! 런치세트 메뉴의 가성비가 미쳤기 때문 .. ㅎㅎ "
          "저는 2인 아건세트로 먹었는데 구성이 알차서 좋았어요 ㅎㅎ 커리는 “치킨티카마살라“ 난은 ”갈릭난“ 음료는 ”망고라씨“로 골랐는데 다 실패없었어요 🧡 "
          "난은 플레인난으로 리필이 가능하다는 것도 너무 좋았어요 플레인난도 담백해서 정말 맛있더라구요 😌😌"
          "직원분들도 너무 친절하고 플레이팅이랑 매장 인테리어도 예쁘고!! 너무 좋았습니다 ㅎㅎ")

response = openai.ChatCompletion.create(
    model="ft:gpt-3.5-turbo-1106:personal::8KJRIJPH",
    temperature=0,
    max_tokens=2048,
    messages=[
        {"role": "system", "content": "You are an expert in extracting keywords from the given review on restaurants."},
        {"role": "user", "content": "Extract price, service, taste, atmosphere keywords form the following reviews. "
                                    "'%s'" %review}
    ],
)
print(response["choices"][0]["message"]["content"])
