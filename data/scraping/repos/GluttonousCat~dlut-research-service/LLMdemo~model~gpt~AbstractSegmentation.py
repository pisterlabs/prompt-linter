from openai import OpenAI
import os


def abstractSegmentation(abstract):
    client = OpenAI()
    response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": "You are a helpful model."},
        {"role": "user", "content": "我需要你进行一个文本处理工作。"},
        {"role": "model", "content": "请告诉我你想要进行的文本处理任务是什么。"},
        {"role": "user", "content": "你的任务就是识别并标注这段文字，标注出文本中的研究问题、研究意义、研究方法、研究结果、研究结论等。"
                                    "但需要注意的是，如果文字中没有表示研究方法等，则无需标注"},
        {"role": "model",
         "content": "好的，我会帮助你识别并标注出文本中的研究意义、研究方法等。请提供你想处理的文本。"},
        {"role": "user",
         "content": "你只需要回答标注结果并分条列出。我将给你一个结果示例格式。如下：1. 研究意义：\n- 科学知识承载于科学发现中，是无形的、复杂的、动态变化的，其发展路径难以被准确测度。\n- "
                    "由于信息的泛滥、知识的局限、人为删除或隐藏等原因，科学发现之间存在较为普遍的、至今尚未被发现的隐性关联。\n2. 研究方法：- "
                    "提出了一种科学发现知识关联方法，结合引文关系中的显性关联（直接引文）与隐性关联（引文耦合、多阶链式）。\n3. 研究结果：\n- "
                    "这种方法能够增强科学发现之间的语义关联，建立更加富有连通性的知识网络。\n- 发现了科学发现的知识传递模式，包括直线模式（简单线性和闭合线性）、桥接模式、放射模式和多重模式。\n4. "
                    "研究结论：\n- 本文的研究结果可以加深科研工作者对领域科学发现规律的理解。可为科研资助、科技评价等政策制定提供参考。"},
        {"role": "model", "content": "好的，我已经记住了你提供的结果示例格式，我会对文本进行标注，并分条列出。请给出文本"},
        {"role": "user", "content": "处理文本如下" + abstract}
      ]
    )
    return response.choices[0].message.content
