# 랭체인
from langchain.llms import OpenAI

# 데이터프레임
import pandas as pd

# JSON으로 변환
import json


def mkdiary(req):
    # req 포맷 : json
    # {
    # 		"ConsumptionLimit": 30000,
    # 		"ConsumptionDetails": {
    # 					"순대국밥": [9000, 3],
    # 					"메가커피": [2000, 4],
    # 		}
    # }

    # ret = {
    #     "content": "오늘의 먹방 대모험, 오늘은 여러 군데에서 맛있는 것들을 먹어봤어! 먼저 서브웨이에서 5900원을 쓰고 먹었는데, 맛이 별로였어. 그 다음엔 매머드커피에서 2000원을 주고 뭔가를 먹었어, 그건 괜찮았단다! 그리고 바나프레소에서 2600원을 주고 먹었는데, 그것도 별로였어. 마지막으로 BBQ치킨에서 완전 대박이었어! 29000원을 주고 치킨을 먹었는데, 그건 정말 대만족!",
    #     "comment ": "우와, 너 정말 많이 먹었네! 근데, 예산이 10000원이라고 했잖아? 너무 많이 초과했어. 다음에는 예산 안에서 먹을 수 있는 맛있는 걸 찾아보자!",
    #     "stamp_type": 1
    # }
    ret = {
        "title": "제목 예시",
        "content": "오늘은 그냥 잠만 잤다",  # text
        "comment ": "참 잘했어요",  # text
        "stampType": 5  # int
    }

    # tempurature : 0 ~ 1 로 높아질수록 랜덤한 답변 생성 / 창의력
    # llm = OpenAI(temperature=0.7)
    llm = OpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.5)

    # 받아온 req로 res_plain_txt 수정
    # 접근하기 쉽게 데이터프레임화
    # BaseModel로 col을 명시해서 col_name와도 문제가 없습니다
    req = json.loads(req.json())

    req_cols = list(req.keys())
    req_limit_amount = pd.DataFrame([req[req_cols[0]]])  # []로 감싼 이유_error: scalar values사용시 인덱스를 써주거나 list로 래핑
    req_consume_list = pd.DataFrame(req[req_cols[1]])

    limit_amount = req_limit_amount.iloc[0, 0]  # 목표 금액
    # limit_amount = req_limit_amount["금액"][0]

    # feeling = ["매우 불만족","불만족","보통","만족","매우 만족"]
    feeling = ["Shit", "Dissatisfied", "Neutral", "Satisfied", "Stoked"]

    consume_list = "[Consume List]\n{"
    for i in req_consume_list.columns:
        consume_one = str(i) + ":" + str(req_consume_list[i][0]) + "Won " + str(
            feeling[(req_consume_list[i][1] - 1)]) + "\n"
        consume_list += consume_one
    consume_list += "}\n"

    # 소비 내역으로 instructions 생성
    # 고정적인 입력값
    # 영어 지시문 혹은 한국어 지시문 둘 다 작성해두었습니다
    instructions = """
        - Compose a diary entry in Korean by referring to the 'Response Format' and 'ConsumeFormat' provided below.
        - Ensure that you strictly adhere to the 'Response Format' which includes keys such as "title", "content", "comment", and "stampType". Do not provide responses that deviate from this format.

        [ConsumeFormat]
        {
            "today consumption limit": amount,
            "today consumption details": {
                "consumer item": [amount, satisfaction(1~5)]
            }
        }

        [ResponseFormat]
        {
            "title": "Provide a funny title with more than 10 characters",
            "content": "Write a delightful diary content, ranging between 50 and 250 characters.",
            "comment": "Based on the 'ConsumeFormat', analyze the spending patterns. Discuss if the consumption was within the set limit, identify any items that might be considered excessive or unnecessary, and suggest possible areas for financial improvement. The content should be between 30 and 80 characters. 존댓말로 작성",
            "stampType": "Rate the spending details on a scale of 1 to 5 (integer)"
        }

    """

    limit_amount_txt = "\nSpendingLimit: " + str(limit_amount) + "Won\n"

    # 최종적으로 GPT에 입력할 텍스트
    input_txt = instructions + consume_list + limit_amount_txt
    res_plain_txt = llm(input_txt)

    # OPEN AI로부터 res를 원하는 형식으로 못 받을 경우 에러 발생
    # ret["title"]
    # ret["content"]
    # ret["comment"]
    # ret["stampType"]

    print(res_plain_txt)
    print("res_plain_txt 끝")

    # try:
    #     ret = json.loads(res_plain_txt)

    # except:
    #     ret = {
    #         "title": "오늘의 먹방 대모험",
    #         "content": "오늘은 여러 군데에서 맛있는 것들을 먹어봤어! 먼저 서브웨이에서 5900원을 쓰고 먹었는데, 맛이 별로였어. 그 다음엔 매머드커피에서 2000원을 주고 뭔가를 먹었어, 그건 괜찮았단다! 그리고 바나프레소에서 2600원을 주고 먹었는데, 그것도 별로였어. 마지막으로 BBQ치킨에서 완전 대박이었어! 29000원을 주고 치킨을 먹었는데, 그건 정말 대만족!",
    #         "comment ": "우와, 너 정말 많이 먹었네! 근데, 예산이 10000원이라고 했잖아? 너무 많이 초과했어. 다음에는 예산 안에서 먹을 수 있는 맛있는 걸 찾아보자!",
    #         "stampType": 1
    #     }

    res_plain_txt = json.loads(res_plain_txt)

    return res_plain_txt

    # food : 식비
    # traffic : 교통
    # online : 온라인 쇼핑
    # offline : 오프라인 쇼핑
    # cafe : 카페/간식
    # housing : 주거/통신
    # fashion : 패션/미용
    # culture : 문화/여가


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("apikey")

    my_req = {
        "ConsumptionLimit": 30000,
        "ConsumptionDetails": {
            "순대국밥": [9000, 3],
            "메가커피": [2000, 4],
        }
    }

    my_request = json.dumps(my_req)
    ret = mkdiary(my_req)

    print(ret)
    # {
    #     "title": "오늘은 맛있는 순대국밥을 먹었어요!",
    #     "content": "오늘은 친구들과 함께 맛있는 순대국밥을 먹었어요. 순대와 국밥이 너무 맛있어서 배가 부르게 먹었어요. 그리고 저녁에는 메가커피도 마셨어요. 커피는 맛있고 가격도 저렴해서 만족스러웠어요.",
    #     "comment": "오늘은 맛있는 음식을 많이 먹었네요! 순대국밥과 메가커피 모두 만족스러웠던 것 같아요. 다만, 소비한 금액이 오늘의 소비 한도를 넘어섰으니 조심해야 할 것 같아요.",
    #     "stampType": 4
    # }
    print("ret 끝")

    print("title: "+ ret["title"])
    print("content: "+ ret["content"])
    print("comment: "+ ret["comment"])
    print("stampType: "+ str(ret["stampType"]))
# title: 오늘은 맛있는 순대국밥을 먹었어요!
# content: 오늘은 친구들과 함께 맛있는 순대국밥을 먹었어요. 순대와 국밥이 너무 맛있어서 배가 부르게 먹었어요. 그리고 저녁에는 메가커피도 마셨어요. 커피는 맛있고 가격도 저렴해서 만족스러웠어요.
# comment: 오늘은 맛있는 음식을 많이 먹었네요! 순대국밥과 메가커피 모두 만족스러웠던 것 같아요. 다만, 소비한 금액이 오늘의 소비 한도를 넘어섰으니 조심해야 할 것 같아요.
# stampType: 4
