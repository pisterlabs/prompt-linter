import openai


class DialogEventGenerator:
    def __init__(self, model_name="gpt3_5", temperature=0.6):
        self.model_name = model_name
        if model_name == "gpt3_5":
            self.core_model = 'gpt-3.5-turbo-0301'
        else:
            raise AttributeError("分类器模型选择参数出错！传入的参数为", model_name)

        self.temperature = temperature
        self.prompt = "用连贯的一句话描述以下对话存在的关键信息，要求包含时间、地点、人物（如果时间和地点不明确，则不需要提出来）：" \
                      "“{{{DIALOG}}}”"

    def send(self, massages):
        return openai.ChatCompletion.create(
            model=self.core_model,
            messages=massages,
            temperature=self.temperature
        )

    def do(self, dialog_str):
        prompt = self.prompt.replace("{{{DIALOG}}}", dialog_str)
        if self.model_name == "gpt3_5":
            massages = [{"role": 'user', "content": prompt}]
            res = self.send(massages)
            ans = res.choices[0].message.content
        else:
            raise AttributeError("分类器模型选择参数出错！传入的参数为", self.model_name)
        return ans
