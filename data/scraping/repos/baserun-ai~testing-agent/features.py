from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI, ChatAnthropic


def choose_llm(provider: str, use_streaming: bool = False):
    if provider == "openai":
        if use_streaming:
            return ChatOpenAI(
                temperature=0,
                model_name="gpt-4-1106-preview",
                streaming=True,
                callbacks=[StreamingStdOutCallbackHandler()],
            )
        else:
            return ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k-0613")
    if provider == "anthropic":
        return ChatAnthropic(streaming=use_streaming)

    raise NotImplementedError
