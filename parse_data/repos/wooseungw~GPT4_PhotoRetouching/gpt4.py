import base64
from openai import OpenAI
import re

class Gpt4:
    def __init__(self, key) -> None:
        self.key = key
        self.client = OpenAI(api_key = key)

    def parse(self, filepath, sentence):
        base64_img = self._img_to_base64(filepath) # 저장된 이미지 파일을 불러와서 base64형식으로 변환
        ex_img = self._img_to_base64('./example_img/ex1.png')
        SYSTEM_INPUT='''
        "역활 놀이를 하자 나는 요청자고 너는 Setimental한 Photo Retouching으로 상을 받은 사진가야.
        나는 내가 원하는 색감이나 감성을 이미지와 함께 입력할꺼야.
        너는 그 이미지의 어떤 오브젝트가 있는지와 광량, 추정된 위치 등을 바탕으로, 피사체를 강조하는게 좋을지 아니면 전체적인 분위기를 보여주는게 좋을지 말해주고,그런데 [text]로 시작해서[/text]로 끝내줘.
        내가 보내준 사진에 어울리는 사진보정방법을 알려줘야해.
        사진 보정법을 알려줄땐 [retouch]로시작해서 [/retouch]로 끝내야해.
        그리고 안에 사진 보정 속성(밝기,대비,노이즈,색상,채도,명도,사프니스,블러) 중 바꿔야 할 부분을 아래의 설명들을 보고 적절한 값을 알려줘.
        그런데 너무 큰값으로고치면 이미지를 알아보기 어려울꺼야.
        밝기: -1~1 사이의 값을 가집니다. 양수면 밝아지고, 음수면 어두워집니다. 기준 0
        대비: 0~2 사이의 값을 가집니다. 1보다 크면 대비가 강해지고, 1보다 작으면 대비가 약해집니다. 기준 1
        노이즈: 0~1 사이의 값을 가집니다. 값이 클수록 노이즈 감소 효과가 강해집니다. 
        색상: 0~2 사이의 값을 가집니다. 값이 클수록 색조가 바뀝니다. 기준 1
        채도: 0~2 사이의 값을 가집니다. 값이 클수록 색상이 선명해집니다. 기준 1
        명도: 0~2 사이의 값을 가집니다. 값이 클수록 밝기가 증가합니다. 기준 1
        선명도: . 0~2 사이의 값을 가집니다. 값이 클수록 선명해집니다. 기준 1
        블러:  0~1 사이의 값을 가집니다. 값이 클수록 블러 효과가 강해집니다. 
        하지만 너무 큰값을 바꿀 필요는 없어. 특히 색상과 채도
        다른 설명은 필요없어.
        예시로 [retouch]밝기: 0.79, 채도: 0.1, 대비: 1.0982, ... [/retouch]
        다 알려주면 대화 종료야. 출력은 소수점 4자리까지 괜찮아 항상 내가 말한 출력폼을 유지해줘"
        '''
        #역할 지정
        SYSTEM_MESSAGE={
            "role": "system",
            "content": [
                {"type": "text", "text": SYSTEM_INPUT},
            ],
        }
        #입력 예시
        PROMPT_EX1 = {
            "role": "user",
            "content": [
                {"type": "text", "text": "좀더 파스텔톤 느낌, 동화같은 느낌으로 만들고싶어"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ex_img}", "detail":"low"}},
            ],
        }
        #출력 예시
        PROMPT_RESULT1= {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "\
                 [text]이 이미지에는 가로등이 보여지며, 그 배경으로 푸르른 나무들과 햇볕이 잘 드는 공원이 포함되어 있어요. 거대한 가로등은 이미지에서 중요한 오브젝트로 보이며, 평온하고 동화 같은 분위기를 연출하기에 적합해 보입니다. 전체적인 분위기를 파스텔톤과 동화같은 느낌으로 만드는 것이 좋겠습니다.[/text]\
                 [retouch]밝기: 0.1, 대비: 0.9, 노이즈: 0.2, 색상: 1.089, 채도: 1.125, 명도: 1.155, 선명도: 1.32, 블러: 0.001[/retouch]\
                 "},
            ],
        }
        #유저 입력
        PROMPT_MESSAGES ={
            "role": "user",
            "content": [
                {"type": "text", "text": sentence},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}", "detail":"low"}},
            ],
        }

        params = {
            "model": "gpt-4-vision-preview",
            "messages": [SYSTEM_MESSAGE, PROMPT_EX1, PROMPT_RESULT1, PROMPT_MESSAGES],
            "max_tokens": 500,
        }

        response = self.client.chat.completions.create(**params)
        res_message = response.choices[0].message.content

        return self._split_sentence_by_keyword(res_message, ['text', 'retouch'])

    def _img_to_base64(self, filepath) -> str:
        '''
        이미지 파일을 filepath에서 불러와서 base64(string) 형식으로 변경
        '''
        with open(filepath, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        
    def _split_sentence_by_keyword(self, sentence: str, keywords: list) -> dict:
        '''
        sentence를 입력받으면 특정 keyword에 따라서 문장을 나눔
        {key1: string, key2: string, ...}
        '''
        result = {}
        for keyword in keywords:
            pattern = fr'\[{keyword}\](.*?)\[/{keyword}\]'
            match = re.search(pattern, sentence, re.DOTALL)
            if match:
                result[keyword] = match.group(1).strip()

        return result
    
    def parse_retouch(retouch):
        retouch_dict = {}
        retouch_items = retouch.split(",")  # 각 기능과 수치를 분리합니다.
        for item in retouch_items:
            key, value = item.split(":")  # 기능과 수치를 분리합니다.
            key = key.strip()
            value = float(value.strip())  # 수치를 실수로 변환합니다.
            retouch_dict[key] = value
        return retouch_dict
