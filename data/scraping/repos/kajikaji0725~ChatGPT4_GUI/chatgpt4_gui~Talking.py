import flet as ft
import openai
import time


class Talking(ft.UserControl):
    def __init__(self):
        super().__init__()
        # self.max_tokens = 2000,
        self.temperature = 0
        self.strem = True
        self.model = 'gpt-3.5-turbo'
        self.is_use_history = False
        self.API_Key = None
        self.is_auto_scroll = False

        self.cl = ft.Column(
            spacing=ft.alignment.center,
            height=100,
            width=200,
            scroll=ft.ScrollMode.ALWAYS,
            auto_scroll=self.is_auto_scroll
        )
        self.ChatHistory = []

    def build(self):
        return ft.Row(controls=[self.cl],
                      alignment=ft.MainAxisAlignment.CENTER,)

    def getter_chatmessage(self):
        return self.ChatHistory

    def getter_config_infomation(self):
        return {"model": self.model, "strem": self.strem, "is_use_history": self.is_use_history, "is_auto_scroll":self.is_auto_scroll, "temperature": self.temperature}

    def getter_API_key(self):
        return self.API_Key

    def getter_cl(self):
        return self.cl

    def setter_config_initialize(self):
        self.temperature = 0
        self.strem = True
        self.model = 'gpt-3.5-turbo'
        self.is_use_history = False
        self.is_auto_scroll = False
        self.cl.auto_scroll = self.is_auto_scroll
        self.setter_histrory_initialize()

    def setter_histrory_initialize(self):
        self.ChatHistory.clear()
        self.cl.clean()
        self.update()

    def setter_stream(self, strem):
        self.strem = strem

    def setter_gpt_model(self, model):
        self.model = model

    def setter_is_use_history(self, is_use_history):
        self.is_use_history = is_use_history
    
    def setter_is_auto_scroll(self,is_auto_scroll):
        self.is_auto_scroll = is_auto_scroll
        self.cl.auto_scroll = self.is_auto_scroll
    
    def setter_temp(self, temp):
        self.temperature = temp

    def setter_API_key(self, apiKey):
        self.API_Key = apiKey
        print(self.API_Key)
        openai.api_key = self.API_Key

    def setter_max_token(self, max_token):
        self.max_tokens = max_token

    def setter_question(self, question):
        self.cl.controls.append(
            ft.Text(f"ユーザ\n{question}\n", size=15, selectable=True))
        message = {"role": "user", "content": question}
        self.ChatHistory.append(message)
        self.update()
        self.setter_answer(message)

    def resize(self, height, width):
        self.cl.height = height - 180
        self.cl.width = width - 200
        self.update()

    def setter_answer(self, question):
        text = ft.Text(size=15,selectable=True)
        self.cl.controls.append(text)
        text.value = 'ChatGPT\n'
        self.update()
        print(question)
        time.sleep(0.5)

        res = openai.ChatCompletion.create(model=self.model,
                                           messages=self.ChatHistory if self.is_use_history else [
                                               question],
                                           temperature=self.temperature,
                                           stream=self.strem)

        # print(self.getter_config_infomation())

        # ans = ""
        # for i in range(10):
        #     text.value += str(i)
        #     ans += str(i)
        #     self.update()

        if self.strem:
            ans = []
            for chunk in res:
                content = chunk["choices"][0]["delta"].get("content", "")
                text.value += str(content)
                self.update()
                ans.append(content)
            ans = "".join(ans).strip()
        else:
            ans = res["choices"][0]["message"]["content"].strip()
            text.value = str(ans)
            u = res["usage"]
            print(u["prompt_tokens"], u["completion_tokens"], u["total_tokens"])
        self.ChatHistory.append({"role": "assistant", "content": ans})

        # self.ChatHistory.append({"role": "assistant", "content": ans})
        self.update()
