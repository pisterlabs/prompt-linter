from langchain import PromptTemplate
from langchain import LLMChain
from llm_wrapper import Baichuan
import streamlit as st
import arxiv
import time
from count import count_everything


st.title("用Baichuan写arxiv最近的论文总结")

def arxiv_search(query):
    summary = {}
    search = arxiv.Search(
      query = query,
      max_results = 1,
      sort_by = arxiv.SortCriterion.SubmittedDate
    )
    
    for result in search.results():
        summary = {"title":result.title,"url":result.entry_id, "abstract":f"title:{result.title},Abstract: {result.summary}, URL: {result.entry_id}"}

        result.download_pdf(dirpath='./papers',filename=result.title.replace(" ","_"))

    return summary


llm = Baichuan()
template = """
    你是一个研究员，为下面一段论文的摘要写总结,
    {abstract},
    总结：
    1、
    2、
    3、
    ...
    \n\n
    """
prompt = PromptTemplate(template=template,input_variables=["abstract"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

def main():
  counter = count_everything()
  query = st.text_input("请用:red[英文]输入要查询的论文关键词")
  if st.button("arxiv论文快报"):
    summary = arxiv_search(query)
  
    try:      
      abstract = summary["abstract"]
      title = summary["title"]
      url = summary["url"]

      start_time = time.perf_counter()
      response = llm_chain.run(abstract)
      last_time = time.perf_counter()- start_time  # 时间单位s
      speed = counter.count_token(response,last_time)

      st.markdown("标题: "+title+"\n\n"+response + "\n\n" + "URL: "+url)
      st.markdown(f":watermelon: :green[{speed}]")
    except KeyError:
       st.markdown("未找到相关论文")


if __name__ == "__main__":
    main()


