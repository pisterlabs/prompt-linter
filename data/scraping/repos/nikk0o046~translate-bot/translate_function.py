def translate(source_language, target_language, source_text):
    from secret_keys import OPENAI_API_KEY
    #import os
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts.chat import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )

    # initialize the OpenAI model
    #chat = ChatOpenAI(temperature=0, openai_api_key = os.getenv('OPENAI_API_KEY'))
    chat = ChatOpenAI(temperature=0, openai_api_key = OPENAI_API_KEY)

    # create the prompt templates
    system_template = """You are a helpful assistant that translates {input_language} to {output_language}.
        The content you translate are srt files. Do not attempt to translate it word by word. Rather, be logical and translate the meaning of the text, keeping the time stamps in place.
        Finally, do not include anything other the translated srt file in your response."""
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    # translate the contents
    openai_response = chat(
        chat_prompt.format_prompt(
            input_language=source_language, output_language=target_language, text=source_text
        ).to_messages()
    )
    translated_contents = openai_response.content

    # return translated contents
    return translated_contents

