import tkinter as tk
from tkinter import scrolledtext
import openai

# 替换成您的 OpenAI API 密钥
api_key = "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"
openai.api_key = api_key

class ChatGPTClient:
    def __init__(self, master):
        self.master = master
        master.title("JGGPT")

        # 问题输入框
        self.question_label = tk.Label(master, text="===请输入你的思路并点击提交===")
        self.question_label.pack()

        self.question_entry = tk.Entry(master, width=140)
        self.question_entry.pack()

        # 间隔
        self.question_label = tk.Label(master, text=" ")
        self.question_label.pack()

        # 显示答案的滚动文本框
        self.answer_text = scrolledtext.ScrolledText(master, width=150, height=40)
        self.answer_text.pack()

        # 间隔
        self.question_label = tk.Label(master, text=" ")
        self.question_label.pack()

        # 提交按钮
        self.submit_button = tk.Button(master, text="提交", command=self.get_answer)
        self.submit_button.pack()

        # 间隔
        self.question_label = tk.Label(master, text=" ")
        self.question_label.pack()

    def get_answer(self):
        # 获取用户输入的问题
        user_question = self.question_entry.get()

        # 等待响应提示
        self.answer_text.insert(tk.END,f"等待响应...\n")

        # 发送问题到 ChatGPT-3.5
        question_prompt = "问题："
        full_prompt = f"{question_prompt} {user_question}"

        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=full_prompt,
            max_tokens=4000,
            n=1,
            stop=None
        )

        # 提取生成的答案
        answer = response.choices[0].text.strip()
        # answer1 = response.choices[1].text.strip()

        # 在滚动文本框中显示答案
        self.answer_text.insert(tk.END, f"问题：{user_question}\n{answer}\n\n------------------------------------------lines------------------------------------------\n")
        self.answer_text.yview(tk.END)

# 创建主窗口
root = tk.Tk()
app = ChatGPTClient(root)

# 运行程序
root.mainloop()
