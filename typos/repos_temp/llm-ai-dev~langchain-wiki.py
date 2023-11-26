"""
例子1: 
=========
已知内容:
问题: golang有哪些优势?

回答: 我不知道

例子2: 
=========   
已知内容:       
Content: 简单的并发
Source: 28-pl
Content: 部署方便
Source: 30-pl

问题: golang有哪些优势?

回答: 部署方便
SOURCES: 28-pl

例子3: 
=========
已知内容:
Content: 部署方便
Source: 0-pl

问题: golang有哪些优势?

回答: 部署方便
SOURCES: 28-pl

例子4:
=========
已知内容:
Content: 简单的并发
Source: 0-pl
Content: 稳定性好
Source: 24-pl
Content: 强大的标准库
Source: 5-pl

问题: golang有哪些优势?

回答: 简单的并发, 稳定性好
SOURCES: 0-pl,24-pl

=========
要求: 1. 参考上面的例子，回答如下问题; 在答案中总是返回 "SOURCES" 信息
要求: 2. 如果你不知道，请说 "抱歉，目前我还没涉及相关知识，无法回答该问题"
要求: 3. 如果你知道，尽可能多的回复用户的问题

已知内容:
{summaries}

问题: {question} 

使用中文回答:  
"""f"""
            {result['answer']}

            **来源：{result['sources']}**
            """