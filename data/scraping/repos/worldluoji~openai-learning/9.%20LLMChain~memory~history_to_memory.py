
from langchain.memory import ConversationSummaryBufferMemory
from langchain import PromptTemplate
from langchain.llms import OpenAI

SUMMARIZER_TEMPLATE = """请将以下内容逐步概括所提供的对话内容，并将新的概括添加到之前的概括中，形成新的概括。

EXAMPLE
Current summary:
Human询问AI对人工智能的看法。AI认为人工智能是一种积极的力量。

New lines of conversation:
Human：为什么你认为人工智能是一种积极的力量？
AI：因为人工智能将帮助人类发挥他们的潜能。

New summary:
Human询问AI对人工智能的看法。AI认为人工智能是一种积极的力量，因为它将帮助人类发挥他们的潜能。
END OF EXAMPLE

Current summary:
{summary}

New lines of conversation:
{new_lines}

New summary:"""

SUMMARY_PROMPT = PromptTemplate(
    input_variables=["summary", "new_lines"], template=SUMMARIZER_TEMPLATE
)

# 可以通过 Memory 提供的 save_context 接口，把历史聊天记录灌进去。然后基于这个 Memory 让 AI 接着和用户对话。比如下面我们就把一组电商客服历史对话记录给了 SummaryBufferMemory
memory = ConversationSummaryBufferMemory(llm=OpenAI(), prompt=SUMMARY_PROMPT, max_token_limit=40)
memory.save_context(
    {"input": "你好"}, 
    {"ouput": "你好，我是客服李四，有什么我可以帮助您的么"}
    )
memory.save_context(
    {"input": "我叫张三，在你们这里下了一张订单，订单号是 2023ABCD，我的邮箱地址是 customer@abc.com，但是这个订单十几天了还没有收到货"}, 
    {"ouput": "好的，您稍等，我先为您查询一下您的订单"}
    )

print(memory.load_memory_variables({}))

# memory.load_memory_variables 方法，我们发现 AI 对整段对话做了小结。但是这个小结有个问题，就是它并没有提取到我们最关注的信息，比如用户的订单号、用户的邮箱