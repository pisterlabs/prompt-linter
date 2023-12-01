from langchain import PromptTemplate
from langchain import LLMChain
from llm_wrapper import Baichuan
import streamlit as st

st.title("新闻编辑工厂")

template = """
    你是一个新闻编辑，根据下面提供的新闻素材，写一篇新闻报道，字数500字以内。需要包含时间和地点。
    素材：{text}
    写一篇新闻报道：
"""

prompt = PromptTemplate(input_variables=["text"],template=template)

llm = Baichuan()
llm_chain = LLMChain(prompt=prompt, llm=llm)

def main():

    text = st.text_area("输入新闻素材")
    if st.button("生成新闻文章"):
        placeholder = st.empty()
        article = llm_chain.run(text)
        placeholder.markdown(article)
        print(article)


if __name__ == "__main__":
    main()