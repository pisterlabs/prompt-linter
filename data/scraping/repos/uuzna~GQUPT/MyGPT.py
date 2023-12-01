import openai


class GPT:
    def __init__(self):
        with open("APIkeys.txt", "r") as file:
            self.api_key = file.readline().strip()
        with open("model_engine.txt", "r") as file:
            self.model_engine = file.readline().strip()

    # 传入一段话,返回结果
    def gpt(self, sentence, temperature = 0.5, max_tokens = None):
        prompt = sentence
        model_engine = self.model_engine
        openai.api_key = self.api_key  #self.api_key

        response = openai.ChatCompletion.create(
            model = model_engine,
            max_tokens = max_tokens,
            n = 1,  # 结果数量
            stop = None,
            temperature = temperature,
            messages = [
            {'role' : 'user', 'content' : prompt},
            ]
        )

        message = response['choices'][0]['message']['content']
        return message

    # input()获取输入,while true循环调用gpt3()
    def auto_chat_gpt(self):
        while True:
            sentence = input('你想要问些什么:')
            message = self.gpt(sentence)
            print(message)


if __name__ == "__main__":
    sentence = '666'
    GPT = GPT()
    GPT.auto_chat_gpt()
    message = GPT.gpt(sentence)
    print(message)
