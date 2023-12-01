import openai


class ModelInteraction:
    def __init__(self, api_key):
        openai.api_key = api_key

    def call_model(self, user_input, functions):
        """
        使用给定的用户输入和函数调用 OpenAI ChatCompletion 模型。

        Args:
            user_input (str): 要传递给模型的用户输入。
            functions (list): 模型要使用的函数列表。

        Returns:
            response: 模型返回的响应对象，如果发生错误，则返回 None。
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=[{"role": "user", "content": user_input}],
                functions=functions
            )
            return response
        except Exception as e:
            print(f"Error in calling the model: {e}")
            return None

    def handle_response(self, response):
        """
        处理从模型收到的响应。

        Args:
            response (dict): 从模型收到的响应。

        Returns:
            str: 从响应中提取的函数调用，如果未找到函数调用，则为 None。
        """
        # 检查是否收到响应
        if response is not None:
            # 检查响应中是否存在选项
            choices = response['choices']
            if choices:
                # 从第一个选项中提取函数调用
                function_call = choices[0]['message']['function_call']
                return function_call
            else:
                # 如果在响应中找不到函数调用，则打印错误消息
                print("No function call found in model's response")
                return None
        else:
            # 如果未收到来自模型的响应，则打印错误消息
            print("No response received from the model")
            return None

    def send_back_result(self, user_input, function_result, functions):
        """
        将函数调用的结果发回给 OpenAI 聊天 API。

        Args:
            user_input (str): 用户的输入。
            function_result (str): 函数调用的结果。
            functions (dict): 自定义函数的字典。

        Returns:
            response (dict): 来自 OpenAI 聊天 API 的响应, 如果出现错误, 则为“None”。
        """
        try:
            # 调用 OpenAI 聊天完成 API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=[
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": None, "function_call": function_result}
                ],
                functions=functions
            )
            return response
        except Exception as e:
            # 如果出现异常，则打印错误消息
            print(f"Error in sending back the result: {e}")
            return None

    def summarize_response(self, response):
        """
        通过从选项中提取消息内容来汇总响应。

        Args:
            response (dict): 从模型收到的响应。

        Returns:
            str or None: 消息内容(如果可用), 否则为“None”。
        """
        if response is not None:
            choices = response['choices']
            if choices:
                message = choices[0]['message']['content']
                return message
            else:
                print("No final message found in model's response")
                return None
        else:
            print("No response received from the model")
            return None
