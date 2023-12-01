import openai

import streamlit as st
from common import write_streaming_response, request_chat_completion

st.title("✍️ AI_카피라이터")
st.subheader("AI를 이용하여 손쉽게 마케팅 문구를 생성해보세요.")
openai.api_key = st.secrets["OPENAI_API_KEY"]


def generate_prompt(product_name, product_desc, num, max_length, keywords):
    prompt = f"""
제품 혹은 브랜드를 SNS에 광고하기 위한 문구를 {num}개 생성해주세요.
자극적이고 창의적으로 작성하세요.
명사 위주로 간결하게 작성하세요.
반드시 {max_length} 단어 이내로 작성해주세요.
키워드가 주어질 경우, 반드시 키워드 중 하나를 포함해야 합니다.
문장을 명사로 끝내세요.
2가지 종류 이상 이모지를 사용하세요.
---
제품명: {product_name}
제품설명: {product_desc}
키워드: {keywords}
---
""".strip()
    return prompt


auto_complete = st.toggle(label="예시로 채우기")
with st.form("form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        example_brand = "카누"
        name = st.text_input(
            label="제품/브랜드 이름(필수)",
            value=example_brand if auto_complete else "",
            placeholder=example_brand
        )
    with col2:
        max_length = st.number_input("최대 단어 수", min_value=5, max_value=20, step=1, value=10)
    with col3:
        num = st.number_input("생성할 문구 수", min_value=1, max_value=10, step=1, value=5)
    example_desc = "집에서도 카페 느낌의 아메리카노 맛이 나는 커피 믹스"
    desc = st.text_input(
        label="제품 간단 정보(필수)",
        value=example_desc if auto_complete else "",
        placeholder=example_desc
    )

    st.text("포함할 키워드(최대 3개까지 허용)")
    col1, col2, col3 = st.columns(3)
    with col1:
        example_keyword_one = "브라질"
        keyword_one = st.text_input(
            label="keyword_1",
            label_visibility="collapsed",
            placeholder=example_keyword_one,
            value=example_keyword_one if auto_complete else ""
        )
    with col2:
        example_keyword_two = "카페"
        keyword_two = st.text_input(
            label="keyword_2",
            label_visibility="collapsed",
            placeholder=example_keyword_two,
            value=example_keyword_two if auto_complete else ""
        )
    with col3:
        example_keyword_three = "공유"
        keyword_three = st.text_input(
            label="keyword_3",
            label_visibility="collapsed",
            placeholder=example_keyword_three,
            value=example_keyword_three if auto_complete else ""
        )
    submitted = st.form_submit_button("제출하기")
if submitted:
    if not name:
        st.error("브랜드 혹은 제품의 이름을 입력해주세요")
    elif not desc:
        st.error("제품의 간단한 정보를 입력해주세요")
    else:
        with st.spinner('AI 카피라이터가 광고 문구를 생성 중입니다...'):
            keywords = [keyword_one, keyword_two, keyword_three]
            keywords = [x for x in keywords if x]
            prompt = generate_prompt(name, desc, num, max_length, keywords)
            system_role = "당신은 전문 카피라이터입니다."
            response = request_chat_completion(prompt, stream=True, system_role=system_role)
        write_streaming_response(response)
