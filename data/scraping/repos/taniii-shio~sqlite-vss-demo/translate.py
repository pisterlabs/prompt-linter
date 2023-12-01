import os
import openai
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def translate(content: str) -> str:
    llm = AzureChatOpenAI(
    deployment_name=os.getenv("CHAT_DEPLOYMENT_NAME"),
    temperature=0,)

    template = """
      次の文章を日本語に翻訳してください。
      文章: {content}
    """

    prompt = PromptTemplate(
        input_variables=["content"],
        template=template
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    result = chain.predict(content=content)
    return result
