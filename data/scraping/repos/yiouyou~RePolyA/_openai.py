from repolya._log import logger_chat


def chat_predict_openai(message, history):
    # print(">>>", message)
    logger_chat.info(message)
    import openai
    history_openai_format = []
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human })
        history_openai_format.append({"role": "assistant", "content":assistant})
    history_openai_format.append({"role": "user", "content": message})
    response = openai.ChatCompletion.create(
        model='gpt-4',
        messages= history_openai_format,         
        temperature=0,
        stream=True
    )
    partial_message = ""
    for chunk in response:
        if len(chunk['choices'][0]['delta']) != 0:
            partial_message = partial_message + chunk['choices'][0]['delta']['content']
            yield partial_message
    # print("<<<", partial_message)
    logger_chat.info(partial_message)

