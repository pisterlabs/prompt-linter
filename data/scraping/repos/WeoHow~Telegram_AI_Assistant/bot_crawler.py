import re

from langchain import LLMChain, PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.chains import LLMRequestsChain

from dotenv import load_dotenv
load_dotenv()

def crawler(text, llm):
    url = re.search(r"(?P<url>https?://[^\s]+)", text, re.UNICODE).group()
    print("偵測到的URL:", url)
    user_require = re.sub(r"\s+", "", text.replace(url, "").strip())  # 去除所有空白字符
    print("URL後面的內容:", user_require)

    template = """
    在 >>> 和 <<< 之間是網頁的返回的HTML的内容。
    請抽取下面要求的信息。

    >>> {requests_result} <<<

    要求:{require}:


    """
    prompt = PromptTemplate(
        template=template,
        input_variables=["requests_result"],
        partial_variables={"require": user_require}
    )

    chain = LLMRequestsChain(llm_chain=LLMChain(llm=llm, prompt=prompt))

    inputs = {
        "url": url,
    }

    with get_openai_callback() as cb:
        response = chain(inputs)
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Successful Requests: {cb.successful_requests}")
        print(f"Total Cost (USD): ${cb.total_cost}")

    return(response)