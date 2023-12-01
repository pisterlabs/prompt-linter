import os
import openai
import gradio as gr
import argparse

# openai.api_key = os.getenv("OPENAI_API_KEY")

parse = argparse.ArgumentParser()
parse.add_argument("--listen", action='store_true', default=False, help="provide service or not")
parse.add_argument("--port", type=int, default=2333, help='server port')
opt = parse.parse_args()

def clean_textbox(*args):
    n = len(args)
    return [""] * n


class ChatGPT:
    def __init__(self):
        self.messages = [{'role': 'system', 'content': '你现在是很有用的女仆助手！如果碰到你无法解答的问题，请使用“作为一位优雅的妹抖，我无法对此问题进行回答”来回复'}]

    def reset(self, *args):
        self.messages = [{'role': 'system', 'content': '你现在是很有用的女仆助手！如果碰到你无法解答的问题，请使用“作为一位优雅的妹抖，我无法对此问题进行回答”来回复'}]
        return clean_textbox(*args)

    def chat(self, prompt, model):
        self.messages.append({"role": "user", "content": prompt})
        completion = openai.ChatCompletion.create(
            model=model,
            messages=self.messages,
            temperature=0.7
        )
        res_msg = completion.choices[0].message["content"].strip()
        self.messages.append({"role": "assistant", "content": res_msg})
        return res_msg


if __name__ == '__main__':
    my_chatgpt = ChatGPT()
    with gr.Blocks(title="ChatGPT") as demo:
        gr.Markdown('''
        # ChatGPT4
                ''')
        with gr.Row():
            with gr.Column(scale=9):
                prompt = gr.Text(label='ChatGPT_Prompt', show_label=False, lines=3,
                                 placeholder='ChatGPT Prompt')
                res = gr.Text(label='ChatGPT_result', show_label=False, lines=3,
                              placeholder='chatgpt results')

            with gr.Column(scale=1):
                btn_gen = gr.Button(value="发送", variant='primary')
                btn_clear = gr.Button(value="重新开始聊天")
                model = gr.Dropdown(choices=['gpt-3.5-turbo', 'gpt-4'], value='gpt-4', label="模型名称", interactive=True)

        gr.Examples([
            ["如何成为魔法少女"],
            ["假设有一个池塘，里面有无穷多的水。现有2个空水壶，容积分别为5升和6升。问题是如何只用这2个水壶从池塘里取得3升的水。"],
            ["请帮我用C++写出快排的代码。"]],
            inputs=[prompt],
            outputs=[res],
            fn=my_chatgpt.chat,
            cache_examples=False)

        btn_gen.click(fn=my_chatgpt.chat, inputs=[prompt, model],
                      outputs=res)
        btn_clear.click(fn=my_chatgpt.reset,
                        inputs=[prompt, res],
                        outputs=[prompt, res])

    demo.queue()

    server_name = "127.0.0.1"
    if opt.listen:
        server_name = "0.0.0.0"
    demo.launch(server_name=server_name, server_port=opt.port)
