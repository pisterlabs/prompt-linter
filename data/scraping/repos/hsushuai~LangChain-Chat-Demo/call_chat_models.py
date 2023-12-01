from langchain.chat_models import QianfanChatEndpoint
from langchain.chat_models import ChatOpenAI
from chat_models import ChatZhipuAI
from dotenv import load_dotenv, find_dotenv


def get_llm(llm_name: str, temperature: float = 0.1):
    """Get large language model.

    To use, you should set your api key in the environment variables.

    Supported llms:
        OPENAI:
            - gpt-3.5-turbo
        ZHIPUAI:
            - chatglm_turbo
        QIANFAN:
            - ERNIE-Bot-turbo.
            you could get the full list from https://cloud.baidu.com/product/wenxinworkshop
        """
    _ = load_dotenv(find_dotenv())
    if llm_name == "chatglm_turbo":
        chat_llm = ChatZhipuAI(temperature=temperature)
    elif llm_name == "gpt-3.5-turbo":
        chat_llm = ChatOpenAI(temperature=temperature)
    elif llm_name == "ERNIE-Bot-turbo":
        chat_llm = QianfanChatEndpoint(temperature=temperature)
    else:
        try:
            chat_llm = QianfanChatEndpoint(model=llm_name, temperature=temperature)
        except ValueError:
            raise ValueError(f"Unsupported language model: {llm_name}")
    return chat_llm


if __name__ == "__main__":
    from langchain.schema.messages import HumanMessage
    message = [HumanMessage(content="你是谁")]
    chat_1 = get_llm(llm_name="chatglm_turbo")
    chat_2 = get_llm(llm_name="gpt-3.5-turbo")
    chat_3 = get_llm(llm_name="ERNIE-Bot-turbo")
    print(
        chat_1(message),
        chat_2(message),
        chat_3(message),
        sep="\n----------\n"
    )
