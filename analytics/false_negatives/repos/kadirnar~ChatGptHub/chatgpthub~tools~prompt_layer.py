def promptlayer_chatgpt(
    text: str = "Hello, I am a chatbot and",
    model_name: str = "gpt-3.5-turbo",
    temperature: float = 0.0,
):
    """
    This function is a template for a chatbot that uses PromptLayer.
    Args:
        text: The text to use as the prompt. Defaults to "Hello, I am a chatbot and".
        model_name: The name of the model to use. Defaults to "gpt-3.5-turbo".
        temperature: The temperature to use for the model. Defaults to 0.0.
    Returns:
        The chatbot's response.
    """

    from langchain.chat_models import PromptLayerChatOpenAI
    from langchain.schema import HumanMessage

    chat = PromptLayerChatOpenAI(
        pl_tags=["langchain"], model_name=model_name, temperature=temperature
    )
    chat_completion = chat([HumanMessage(content=text)]).content

    return chat_completion
