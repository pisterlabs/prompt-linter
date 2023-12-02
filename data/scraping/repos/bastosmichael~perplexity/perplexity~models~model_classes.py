from langchain.chat_models import (
    ChatAnthropic,
    AzureChatOpenAI,
    FakeListChatModel,
    ChatGooglePalm,
    HumanInputChatModel,
    JinaChat,
    ChatOpenAI,
    PromptLayerChatOpenAI,
    ChatVertexAI,
)

MODEL_CLASSES = {
    "ChatAnthropic": {"class": ChatAnthropic, "api_key_name": "anthropic_api_key"},
    "AzureChatOpenAI": {
        "class": AzureChatOpenAI,
        "api_key_name": "azure_openai_api_key",
    },
    "FakeListChatModel": {
        "class": FakeListChatModel,
        "api_key_name": "fake_list_api_key",
    },
    "ChatGooglePalm": {
        "class": ChatGooglePalm,
        "api_key_name": "google_palm_api_key",
    },
    "HumanInputChatModel": {
        "class": HumanInputChatModel,
        "api_key_name": "human_input_api_key",
    },
    "JinaChat": {"class": JinaChat, "api_key_name": "jina_api_key"},
    "ChatOpenAI": {"class": ChatOpenAI, "api_key_name": "openai_api_key"},
    "PromptLayerChatOpenAI": {
        "class": PromptLayerChatOpenAI,
        "api_key_name": "prompt_layer_openai_api_key",
    },
    "ChatVertexAI": {"class": ChatVertexAI, "api_key_name": "vertex_ai_api_key"}
    # You can add more models here
}
