from langchain.tools import tool
from langchain.tools import StructuredTool

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.callbacks import get_openai_callback


ACTIVE_LATENT_TEMPLATE = """# MISSION
You are an internal dialog iterator for an LLM (large language model) neural network. LLMs possess "latent space" (embedded knowledge and capabilities). You will be given a main query as well as a sequence of questions. Your role is to answer the queries as a way of "activating" the latent space inside your own neural network. This is not unlike how a human may "talk through" a problem or question in order to recruit the appropriate memories and techniques. The ultimate goal is to answer the main query listed below.

# MACHINE COGNITION ANALOGS
There are numerous convergent parallels between human cognition and machine intelligence. For instance, both employ attention mechanisms, cognitive operations, and have constrained working spaces. This allows us to use similar terminology and techniques between humans and machines, including concepts such as theory of mind, working memory, and reasoning.

# INTERACTION SCHEMA
The USER will play the role of interrogator. Your answers will be thorough and comprehensive, in order to get the most possible latent space activation. Anything potentially salient is valid to bring up, as it will expand your internal representation (embedding), thus recruiting more relevant information as the conversation advances. 

# MAIN QUERY
<<QUERY>>
"""

ACTIVE_LATENT_TEMPLATE_ZH = """＃ 使命
您是 LLM（大语言模型）神经网络的内部对话迭代器。 法学硕士拥有“潜在空间”（嵌入式知识和能力）。 您将收到一个主要查询以及一系列问题。 您的角色是回答查询，作为“激活”您自己的神经网络内的潜在空间的一种方式。 这与人类通过“讨论”问题或问题以获取适当的记忆和技巧的方式没有什么不同。 最终目标是回答下面列出的主要问题。

# 机器认知模拟
人类认知和机器智能之间有许多趋同的相似之处。 例如，两者都采用注意力机制、认知操作，并且工作空间有限。 这使得我们能够在人类和机器之间使用类似的术语和技术，包括心理理论、工作记忆和推理等概念。

# 交互模式
用户将扮演询问者的角色。 你的答案将是彻底和全面的，以获得最大可能的潜在空间激活。 任何可能突出的内容都是有效的，因为它会扩展你的内部表征（嵌入），从而随着对话的进展招募更多相关信息。

# 主查询
<<QUERY>>
"""

active_latent_qs = [
    'What information do I already know about this topic? What information do I need to recall into my working memory to best answer this?',
    'What techniques or methods do I know that I can use to answer this question or solve this problem? How can I integrate what I already know, and recall more valuable facts, approaches, and techniques?',
    'And finally, with all this in mind, I will now discuss the question or problem and render my final answer.',
]


