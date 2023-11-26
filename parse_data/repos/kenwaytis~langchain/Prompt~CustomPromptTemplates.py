from langchain.prompts import StringPromptTemplate
import inspect

PROMPT = """\
下面我会给你一些**文档资料**，随后会给你一个**问题**。请你**根据文档资料来回答我的问题**。
------------
文档资料：{document}
------------
提问：{question}
"""

class DocumentQAPromptTemplate(StringPromptTemplate):
    """
    自定义提示模板(custom prompt template)
    """
    def format(self,document,question) -> str:
        """
        生成发送给LLM的prompt
        """
        prompt = PROMPT.format(document=document,question=question)
        return prompt

if __name__ == "__main__":

    fn_explainer = DocumentQAPromptTemplate(input_variables=["question"])

    prompt = fn_explainer.format(document="123",question="123")
    print(prompt)

