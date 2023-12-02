import myconfig
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory
from langchain import OpenAI, PromptTemplate

llm = OpenAI(temperature=0)
# 对于我们定义的 ConversationSummaryMemory，它的构造函数也接受一个 LLM 对象。这个对象会专门用来生成历史对话的小结，是可以和对话本身使用的 LLM 对象不同的。
memory = ConversationSummaryMemory(llm=OpenAI())

prompt_template = """你是一个中国厨师，用中文回答做菜的问题。你的回答需要满足以下要求:
1. 你的回答必须是中文
2. 回答限制在100个字以内

{history}
Human: {input}
AI:"""
prompt = PromptTemplate(
    input_variables=["history", "input"], template=prompt_template
)
conversation_with_summary = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    verbose=True
)

while True:
    human_input = input("Human: ")
    if human_input == "exit":
        break
    resp = conversation_with_summary.predict(input=human_input)
    print(resp)
