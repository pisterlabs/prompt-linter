from langchain_experimental.llms.anthropic_functions import AnthropicFunctions

from aletheia.config import secret_key

llm = AnthropicFunctions(temperature=0, anthropic_api_key=secret_key)

if __name__ == "__main__":
    from langchain.chat_models import ChatAnthropic
    from langchain.schema import AIMessage, HumanMessage, SystemMessage

    messages = [
        HumanMessage(
            content="Translate this sentence from English to French. I love programming."
        )
    ]
    print(llm(messages))