# Memory 组件 Demo
# 简单实现一个AI面试后的打分功能

# 导入OpenAi
import os
os.environ["OPENAI_API_KEY"] = "xxx"
from langchain.llms import OpenAI
llm = OpenAI(openai_api_key="xxx")

from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.memory import ConversationSummaryMemory, ChatMessageHistory

# 面试记录，此处直接生成（应用中则是根据对话状态动态添加的）
history = ChatMessageHistory()
history.add_user_message("面试官你好")
history.add_ai_message("你好，我是你的AI面试官。请进行自我介绍")
history.add_user_message("我叫小帅，从事前端开发3年了")
history.add_ai_message("好的，让我们来开始面试第一道题：说说你对css盒模型的理解")
history.add_user_message("不好意思，这块内容不是很了解")
history.add_ai_message("好的下一道题：js有哪些数据类型")
history.add_user_message("有number、string、boolean、Symbol这些吧")
history.add_ai_message("好的面试结束，请等待我们的打分")

# 使用Memory组件处理history
memoryMessage = ConversationSummaryMemory.from_messages(llm=OpenAI(temperature=0), chat_memory=history, return_messages=True)
memory = memoryMessage.buffer

print(memory)

# 自定义的答案知识库
answer = """
    CSS盒模型是指用于布局和渲染HTML元素的一种模型。
    它将每个HTML元素视为一个矩形的盒子，这个盒子包含了元素的内容、内边距、边框和外边距。
    CSS盒模型由以下几个部分组成：
    内容区域（content area）：盒子的实际内容部分，例如文本、图像等。它的大小由元素的width和height属性决定。
    内边距（padding）：内容区域与边框之间的空白区域。可以通过padding属性来设置内边距的大小。
    边框（border）：包围内容区域和内边距的线条。可以通过border属性来设置边框的样式、宽度和颜色。
    外边距（margin）：盒子与周围元素之间的空白区域。可以通过margin属性来设置外边距的大小。
    这些部分相互组合形成一个完整的盒子，它们的大小和样式可以通过CSS来控制和调整。
    CSS盒模型的默认行为是"content-box"，即宽度和高度仅包括内容区域，而不包括内边距、边框和外边距。
    然而，可以通过设置盒模型的box-sizing属性为"border-box"来改变默认行为，使宽度和高度包括内边距和边框。

    JavaScript具有以下基本数据类型：
    布尔类型（Boolean）：表示逻辑上的 true 或 false。
    数字类型（Number）：表示数值，可以是整数或浮点数。
    字符串类型（String）：表示文本数据，用单引号（'）或双引号（"）括起来。
    空值（Null）：表示一个空值或不存在的值。
    未定义（Undefined）：表示一个未定义的值。
    除了上述基本数据类型，JavaScript 还具有以下复杂数据类型：
    对象类型（Object）：表示复杂的数据结构，可以包含多个键值对。
    数组类型（Array）：表示按顺序排列的一组值，可以通过索引访问。
    函数类型（Function）：表示可执行的代码块。
    日期类型（Date）：表示日期和时间。
    正则表达式类型（RegExp）：表示匹配某种模式的文本。
    Symbol类型（Symbol）：表示唯一的标识符，用于对象属性的键。
"""

# 将Memory组件处理过的面试记录和答案丢给AI进行打分
template = """
你是一个前端开发面试官，这是面试者的答案：
{history}

--------------------

这是答案库：
{answer}

--------------------
满分100分，请按照面试者的情况打分
"""

prompt = PromptTemplate(
	input_variables=["history", "answer"],
	template=template,
)
prompt_value = prompt.format(history=memory, answer=answer)
print(prompt_value)
score = llm(prompt_value)
print(score)




