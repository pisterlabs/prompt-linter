from dotenv import load_dotenv
load_dotenv()

# 导入所需的库
from langchain import PromptTemplate, OpenAI, LLMChain
# 原始字符串模板
template = "{flower}的花语是?"
# 创建模型实例
llm = OpenAI(temperature=0)
# 创建Prompt
prompt = PromptTemplate(
    input_variables=["flower", "season"],
    template="{flower}在{season}的花语是?",
)
# 创建LLMChain
llm_chain = LLMChain(llm=llm, prompt=prompt)
# 调用LLMChain，返回结果
print(llm_chain({
    'flower': "玫瑰",
    'season': "夏季" }))

