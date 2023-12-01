from dotenv import load_dotenv
load_dotenv()

from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain

# 这是第一个LLMChain，用于生成鲜花的介绍，输入为花的名称和种类
llm = OpenAI(temperature=.7)
template = """
你是一个植物学家。给定花的名称和类型，你需要为这种花写一个200字左右的介绍。

花名: {name}
颜色: {color}
植物学家: 这是关于上述花的介绍:"""
prompt_template = PromptTemplate(input_variables=["name", "color"], template=template)
introduction_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="introduction")

# 这是第二个LLMChain，用于根据鲜花的介绍写出鲜花的评论
llm = OpenAI(temperature=.7)
template = """
你是一位鲜花评论家。给定一种花的介绍，你需要为这种花写一篇200字左右的评论。

鲜花介绍:
{introduction}
花评人对上述花的评论:"""
prompt_template = PromptTemplate(input_variables=["introduction"], template=template)
review_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="review")


# 这是第三个LLMChain，用于根据鲜花的介绍和评论写出一篇自媒体的文案
template = """
你是一家花店的社交媒体经理。给定一种花的介绍和评论，你需要为这种花写一篇社交媒体的帖子，300字左右。

鲜花介绍:
{introduction}
花评人对上述花的评论:
{review}

社交媒体帖子:
"""
prompt_template = PromptTemplate(input_variables=["introduction", "review"], template=template)
social_post_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="social_post_text")

# 这是总的链，我们按顺序运行这三个链
overall_chain = SequentialChain(
    chains=[introduction_chain, review_chain, social_post_chain],
    input_variables=["name", "color"],
    output_variables=["introduction","review","social_post_text"],
    verbose=True)

# 运行链，并打印结果
result = overall_chain({"name":"玫瑰", "color": "黑色"})
print(result)