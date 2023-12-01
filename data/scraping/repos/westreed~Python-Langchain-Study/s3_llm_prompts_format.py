import os
from key import APIKEY
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate


if __name__ == "__main__":
    KEY = APIKEY()
    llm = OpenAI(openai_api_key=KEY.openai_api_key, temperature=0.9)

    # prompt 텍스트에 포맷팅 기법 쓰는 방법
    prompt = PromptTemplate.from_template("What is a {how} name for a company that makes {product}?")
    formattingPrompt = prompt.format(how="good", product="colorful socks")
    print(formattingPrompt)
    print(llm.predict(formattingPrompt))