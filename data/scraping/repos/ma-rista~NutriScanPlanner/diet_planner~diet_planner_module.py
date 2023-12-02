import gradio as gr
import openai
import deepl
from openai import OpenAI

# OpenAI 라이브러리에 API 키 설정
openai.api_key = ''
client = OpenAI(api_key='')

# DeepL API 인증 키 설정
auth_key = "6309462f-ad40-dba2-f27f-e297c462fcd9:fx"
translator = deepl.Translator(auth_key)

def translate_text_with_deepl(text, target_language="KO"):
    try:
        result = translator.translate_text(text, target_lang=target_language)
        return result.text
    except deepl.DeepLException as error:
        print(error)
        return text  # 번역에 실패한 경우 원문 반환

def generate_diet_plan(calories, ingredients, cuisine, dietary_restrictions, allergies, medical_conditions, meals_per_day, cooking_preference):
    # 채팅 형식의 메시지 생성
    messages = [
        {"role": "system",
         # "content": "Use Markdown formatting to create meal plans. You are a nutrition expert. Your task is to develop meal plans that meet the user's specified dietary needs. Your responses should be detailed, structured, and informative, utilizing Markdown tables to present the meal plan. Make sure to consider the user's calorie goals, preferred ingredients, dietary restrictions, and the number of meals per day. Provide a breakdown of each meal with nutritional information such as calorie content and macronutrients."},
         "content":"식단 계획을 마크다운 형식으로 작성하세요.당신은 영양 전문가입니다.사용자가 지정한 식단 요구 사항을 충족시키는 식단 계획을 개발하는 것이 당신의 임무입니다.답변은 상세하고 구조화되며 유익해야 하며,식단 계획을 제시하는 데 마크다운 표를 사용해야 합니다.사용자의 칼로리 목표, 선호 재료, 식이 제한, 하루 식사 횟수를 고려하세요.각 식사에 대한 분석을 제공하며, 칼로리 함량 및 주요영양소와 같은 영양 정보를 포함시키세요."},
        {"role": "user", "content": f"Create a diet plan with the following requirements:\n{calories}: Your target calorie count for the day.\n{ingredients}: The ingredients that make up your diet (we'll use the best we can, but you're welcome to make other suggestions)\n{cuisine}: Your preferred food style\n{dietary_restrictions}: Food groups you want to limit (dietary restrictions)\n{allergies}: Allergies and intolerances\n{medical_conditions}: Diseases or medical conditions you suffer from.\n{meals_per_day}: Number of meals you want to eat per day\n{cooking_preference}: Preferred cooking time."},
        {"role": "assistant", "content": f"""
                키와 체중을 고려하여 열량을 조절하고, 단백질 섭취량을 100~120g으로 맞추기 위해 식단을 조절하겠습니다. 아래는 조정된 식단의 예시입니다. 실제 식단의 세부 사항은 각 음식의 크기, 조리 방법에 따라 달라질 수 있습니다.
                |식사    |음식    |양    |열량 (kcal) |
                |----|---|---|----|
                아침 식사|스크램블 에그와 야채    |2개, 야채 추가|300|18|

                **합계**

                - 열량: 약 2200 kcal
                - 단백질: 100~120g (변동 가능)
                """
         },
        # 추가 사용자 및 어시스턴트 메시지가 필요한 경우 여기에 포함시킵니다.
    ]

    # GPT API 호출
    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages
    )

    # 결과를 마크다운 형식으로 변환
    diet_plan = completion.choices[0].message.content
    # translated_diet_plan = translate_text_with_deepl(diet_plan, "KO")
    # markdown_format = f"# 생성된 식단 계획\n\n{translated_diet_plan}"
    # markdown_format = f"# 생성된 식단 계획\n\n{diet_plan}"

    return gr.Markdown(value = diet_plan)

# Gradio 인터페이스 정의 함수
def create_diet_planner_interface():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                # 입력 컴포넌트 구성
                calories = gr.Number(label="TDEE 계산기로 입력받은 칼로리")
                ingredients = gr.Textbox(label="식재료")
                cuisine = gr.CheckboxGroup(choices=["한식", "중식", "양식"], label="카테고리")
                dietary_restrictions = gr.CheckboxGroup(choices=["채식", "저탄수화물"], label="식이 제한")
                allergies = gr.CheckboxGroup(choices=["땅콩", "우유", "글루텐"], label="알레르기 및 불내성")
                medical_conditions = gr.CheckboxGroup(choices=["당뇨병", "고혈압"], label="의료 상태")
                meals_per_day = gr.Radio(choices=["2끼", "3끼", "4끼"], label="하루 몇 끼 섭취")
                cooking_preference = gr.CheckboxGroup(choices=["간단한 조리", "긴 조리 시간"], label="조리 시간 및 용이성")
                submit_button = gr.Button("식단 생성")

            with gr.Column():
                # 결과 출력
                result = gr.Markdown()

            submit_button.click(
                generate_diet_plan,
                inputs=[calories, ingredients, cuisine, dietary_restrictions, allergies, medical_conditions, meals_per_day, cooking_preference],
                outputs=result
            )

    return demo, translate_text_with_deepl, generate_diet_plan

# 인터페이스 생성 함수 호출
diet_planner_interface = create_diet_planner_interface()
