"""
	链式结构
		WHY?
			连接多个 llm 模块
   		
    提示词
     	如何避免重复定义功能相似的 llm 模块？
			提示模板 {}
					prompt template
     
		提供简单的示例

		给出限制
"""
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


# 用户的输入
def get_user_input():
	user_input = input("请输入对象（例如'小孩'、'科技公司'等）：")
	return user_input


# 大模型的运行
def run_llm(user_input): 
	llm = OpenAI() # llm = OpenAI(openai_api_key='...')
	prompt = PromptTemplate.from_template("帮我给{placeholder}起一个很酷的名称")
	prompt.format(placeholder=user_input) # "帮我给科技公司起一个很酷的名称" => 只是一个模板的例子, 实际还是 👇 的输入
	#  prompt.format(对象="科技公司") # "帮我给科技公司起一个很酷的名称" => 只是一个模板的例子, 实际还是 👇 的输入

	chain = LLMChain(llm=llm, prompt=prompt)
	res = chain.run(user_input) # 🌟放入用户输入的问题
	return res


# 主程序
if __name__ == "__main__":
    user_input = get_user_input()
    result = run_llm(user_input)
    print(result)

