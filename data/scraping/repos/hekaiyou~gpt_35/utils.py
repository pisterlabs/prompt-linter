import openai


def gpt_35_api(api_key: str, messages: list):
    """为提供的对话消息创建新的回答

    Args:
        messages (list): 完整的对话消息
        api_key (str): OpenAI API 密钥

    Returns:
        tuple: (results, error_desc)
    """
    try:
        openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=messages,
        )
        messages.append({
            'role':
            response['choices'][0]['message']['role'],
            'content':
            response['choices'][0]['message']['content'],
        })
        return (True, '')
    except Exception as err:
        return (False, f'OpenAI API 异常: {err}')


def gpt_35_api_stream(api_key: str, messages: list):
    """为提供的对话消息创建新的回答 (流式传输)

    Args:
        messages (list): 完整的对话消息
        api_key (str): OpenAI API 密钥

    Returns:
        tuple: (results, error_desc, response)
    """
    try:
        openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=messages,
            stream=True,
        )  # <class 'generator'> 流式传输的生成器
        return (True, '', response)
    except Exception as err:
        return (False, f'OpenAI API 异常: {err}', None)
