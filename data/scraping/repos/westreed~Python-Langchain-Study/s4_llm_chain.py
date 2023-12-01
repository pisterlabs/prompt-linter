from key import APIKEY
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


if __name__ == "__main__":
    KEY = APIKEY()
    llm = OpenAI(openai_api_key=KEY.openai_api_key, temperature=0.9)

    prompt = PromptTemplate.from_template("What is a {how} name for a company that makes {product}?")

    # 위에서 생성한 prompt와 llm을 chain으로 엮고, run에서 prompt인자를 넣을 수 있음.
    chain = LLMChain(llm=llm, prompt=prompt)
    print(chain.run(how="bad", product="colorful socks"))