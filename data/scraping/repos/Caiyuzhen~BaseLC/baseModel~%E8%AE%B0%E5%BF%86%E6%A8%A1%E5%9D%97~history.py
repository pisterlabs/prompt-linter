"""
	历史记录模块
		What?
			保留对话的上下文
		What?
			1.用于 chain 、agent 等结构
			2.可以对 memory 进行修改
			3.可以利用数据库对历史记录进行存储, 比如 mongoDB 或 AWS
"""


from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI


# 创建 OpenAI 语言模型的实例
llm = OpenAI()


# 定义模板，它将包含聊天历史和人类的输入
template = """
	你是一个聊天 BOT, 能够保留聊天的上下文。
 
 	{chat_history}
  
    人类:{human_input}
    BOT:
"""

# 创建 PromptTemplate 实例

prompt = PromptTemplate(
	input_variables=["chat_history", "human_input"], 
 	template=template
)


# 初始化聊天历史
chat_history = ""


# 模拟一些人类输入
# human_inputs = ["你是我的小猫叫嘟嘟！", "你会喵喵叫", "你可以帮我拿外卖!"]
human_inputs = [""] # 🔥可以把用户的输入存为历史记录或者【🌟存入数据库】！


# 对于每个人类输入，生成并打印机器人的回复
# 循环，直到用户选择【停止】
while True:
    # 🌟获取用户的输入
    human_input = input("请输入您的问题（输入'退出'以结束对话）: ")
    
    # 检查是否退出
    if human_input.lower() == '退出':
        break
    
    
    # 1.更新聊天历史以包括新的人类输入 ｜ 2.生成新的回复
    chat_history += f"人类:{human_input}\n" 
	
	# 🌟【带有历史记录的提示词！】使用模板和输入生成提示
    generated_prompt = prompt.format(
		chat_history=chat_history, 
		human_input=human_input
	)
	
	# 使用语言模型生成回复
    res = llm.generate([generated_prompt]) # 🌟 包装成列表
	
	# 从响应中提取文本
    bot_response_text = res.generations[0][0].text  # 提取第一个 Generation 对象的文本
	
	# 为下一个机器人的循环回复添加到聊天历史中
    chat_history += f"BOT:{bot_response_text}\n"

	# 打印机器人的回复
    print(f"BOT:{bot_response_text}")