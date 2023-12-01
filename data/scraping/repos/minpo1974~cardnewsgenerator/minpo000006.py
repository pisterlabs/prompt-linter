import streamlit as st
import os
from langchain.chat_models import ChatOpenAI

def generate_card_news(topic):
    # 전문적인 프롬프트 작성
    professional_prompt = f"카드뉴스 생성: {topic}"
    # 모델을 사용하여 카드뉴스 컨텐츠 생성
    chat_model = ChatOpenAI()
    result_chat_model = chat_model.predict(professional_prompt)
    return result_chat_model

# Streamlit 앱
st.title("영산대학교 카드뉴스 생성 서비스")

# 사용자로부터 API 키 및 카드뉴스 주제 입력받기
api_key = st.text_input("OpenAI API 키를 입력하세요:", type="password")
card_news_topic = st.text_input("카드뉴스 주제를 입력하세요:")

if api_key and card_news_topic:
    # 환경 변수 설정
    os.environ['OPENAI_API_KEY'] = api_key
    # 카드뉴스 생성 및 출력
    card_news_content = generate_card_news(card_news_topic)
    st.write(card_news_content)
