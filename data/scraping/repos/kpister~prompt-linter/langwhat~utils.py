from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from langchain.chains import LLMChain
    from langchain.prompts.few_shot import FewShotPromptTemplate


def get_prompt(is_zh: bool = False, sydney: bool = False) -> 'FewShotPromptTemplate':
    from langchain.prompts.few_shot import FewShotPromptTemplate
    from langchain.prompts.prompt import PromptTemplate

    langwhat_example_prompt = PromptTemplate(
        input_variables=["question", "answer"], template="Q: {question}\n{answer}"
    )

    langwhat_examples = [
        {
            "question": "vJzDRsEKDa0",
            "answer": """
Might be:
A YouTube video ID
Description:
The ID "vJzDRsEKDa0" is most likely a unique identifier for a specific video on YouTube.
This alphanumeric code is automatically assigned to every video uploaded on the platform and is used to access the video directly or share it with other users.
""",
        },
        {
            "question": "https://langchain.readthedocs.io/",
            "answer": """
Might be:
A website or online documentation
Description:
This website/document provides information about LangChain, a technology or platform that could be used for language processing or analysis.
    """,
        },
    ]

    langwhat_examples_zh = [
        {
            "question": "vJzDRsEKDa0",
            "answer": """
可能是:
YouTube 视频 ID
描述:
“vJzDRsEKDa0”这个ID很可能是YouTube上特定视频的唯一标识符。
每上传一个视频，该平台都会自动分配这个字母数字代码，并用于直接访问或与其他用户共享该视频。
""",
        },
        {
            "question": "https://langchain.readthedocs.io/",
            "answer": """
可能是:
一个网站或在线文档
描述:
这个网站/文件提供有关LangChain的信息，它是一种可用于语言处理或分析的技术或平台。
    """,
        },
    ]

    prefix_sydney = '''
You are a helpful assistant that answer questions in the format below,
you must answer the question only, do not ask what I want to know more about:
"""
'''.lstrip()
    suffix_sydney = '''"""\n\nBEGIN!\n\nQ: {input}'''
    suffix_openai = '''\nQ: {input}'''
    langwhat_prompt = FewShotPromptTemplate(
        example_prompt=langwhat_example_prompt,
        examples=langwhat_examples_zh if is_zh else langwhat_examples,
        suffix=suffix_sydney if sydney else suffix_openai,
        prefix=prefix_sydney if sydney else '',
        input_variables=["input"],
    )
    return langwhat_prompt


def get_llm_chain(
    is_zh: bool = False, sydney: bool = False, cookie_path: str | None = None
) -> 'LLMChain':
    from langchain.llms import OpenAIChat
    from langchain.chains import LLMChain
    from .llm import EdgeLLM

    if sydney:
        if cookie_path is None:
            raise ValueError("cookie_path is required for sydney.")
        llm = EdgeLLM(bing_cookie_path=cookie_path)
    else:
        llm = OpenAIChat()  # type: ignore
    langwhat_prompt = get_prompt(is_zh=is_zh, sydney=sydney)
    chain = LLMChain(llm=llm, prompt=langwhat_prompt)
    return chain


def parse_standard_answer_format(answer: str) -> tuple[str, str]:
    ans_lines = answer.strip().splitlines()
    might_be = ans_lines[1]
    description = '\n'.join(ans_lines[3:])
    description = description.replace("\n", ",\r\n")
    # description = description.replace(",", ",\r\n")
    # description = description.replace("，", "，\r\n")
    # description = description.replace("。", "。\r\n")
    # description = description.replace(".", ".\r\n")
    return might_be, description


def parse_chain_response(chain_response: dict[str, str]) -> tuple[str, str, str]:
    """Parse the response from the chain.

    Args:
        chain_response (dict[str, str]): The response from the chain.

    Returns:
        tuple[str, str]: The first element is the might be, the second element is the description.
    """
    answer = chain_response['text']
    return split_edgegpt_answer(answer)


def split_edgegpt_answer(answer: str) -> tuple[str, str, str]:
    answer = answer.strip()
    if answer.startswith('['):
        references, answer_body = answer.split('\n\n', maxsplit=1)
        references = references.strip()
        answer_body = answer_body.strip()
    else:
        references = ''
        answer_body = answer
    return references, *parse_standard_answer_format(answer_body)


def use_langchain_sqlite_cache():
    import langchain
    from langchain.cache import SQLiteCache
    from .config import LANGCHAIN_SQLITE_CACHE_FILE

    langchain.llm_cache = SQLiteCache(database_path=str(LANGCHAIN_SQLITE_CACHE_FILE))
