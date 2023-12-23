def translate_chatgpt(
    model_name: str = "gpt-3.5-turbo",
    input_language: str = "English",
    output_language: str = "Turkish",
    text: str = "Hello, how are you?",
    temperature: float = 0.0,
):
    """
    This function is a template for a chatbot that translates between two languages.
    Args:
        model_name: The name of the model to use. Defaults to "gpt-3.5-turbo".
        input_language: The language to translate from. Defaults to "English".
        output_language: The language to translate to. Defaults to "Turkish".
        text: The text to translate. Defaults to "Hello, how are you?".
        temperature: The temperature to use for the model. Defaults to 0.0.
    Returns:
        The translated text.
    """
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts.chat import (
        ChatPromptTemplate,
        HumanMessagePromptTemplate,
        SystemMessagePromptTemplate,
    )

    chat = ChatOpenAI(model_name=model_name, temperature=temperature)
    template = "You are a helpful assistant that translates {input_language} to {output_language}."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(
        human_template
    )

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chat_completion = chat(
        chat_prompt.format_prompt(
            input_language=input_language,
            output_language=output_language,
            text=text,
        ).to_messages()
    )
    last_ai_message = chat_completion.content

    return last_ai_message
