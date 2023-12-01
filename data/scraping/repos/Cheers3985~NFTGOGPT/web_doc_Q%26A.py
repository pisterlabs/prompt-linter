from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import WebBaseLoader
from langchain import OpenAI
from langchain.prompts import PromptTemplate
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter.split_text()
text_splitter = RecursiveCharacterTextSplitter.split_documents()
# WebBaseLoader 加载网页可视化的文字内容。
# 构建自己的标签系统来进行打标签处理操作
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_PROXY"] = "http://127.0.0.1:10809"
prompt_template ="I would like you to create an article summary. I will provide you with an article on a specific topic,\
and you will summarize the main points of the article. The summary should be concise and accurately \
convey the main points of the article without including personal opinions or interpretations. \
The focus is on objectively presenting the information in the article, rather than adding your own views.\
The summary should be expressed in your own words and not directly quote the article.\
Please ensure that the summary is clear, concise, and accurately reflects the content of the article.the article will \
Summarize the review below, delimited by triple\
Review: {text}\
CONCISE SUMMARY IN CHINESE:\
"
llm = OpenAI(temperature=0,model_name='gpt-3.5-turbo-0613')

# loader = WebBaseLoader("https://zhuanlan.zhihu.com/p/617487439")
def summary_article(url):
    loader = WebBaseLoader(url)
    docs  = loader.load()
    # 去除空格和换行符
    # 不需要再对document进行处理了，直接使用chain来对其进行总结，不要再对其转换成字符串形式。
    # data = data[0].page_content.strip().replace("\n","")  # 获取page_content    print(data)
    # 自定义中input_varbles= ["text"] 必须是这个，不能更改其他的，对于文档类型来说
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    # data = "Using this allows you to track the performance of your model in the PromptLayer dashboard. If you are using a prompt template, you can attach a template to a request as well. Overall, this gives you the opportunity to track the performance of different templates and models in the PromptLayer dashboard."
    chain = load_summarize_chain(llm=llm,prompt=PROMPT,chain_type='stuff')
    summary = chain.run(docs)
    # summary = chain.predict(text=data)
    return summary
if __name__ == '__main__':
    summary = summary_article("https://blog.langchain.dev/langchain-vectara-better-together/")
    print(summary)
# 新增总结文章的内容

# 新增评分文章的内容

# 新增导出的内容

