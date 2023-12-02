import openai
import gradio

openai.api_key = "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"

messages = [{"role": "system", "content": "Marv is a informational guide and an expert on computer science and Java programming."}]

def CustomChatGPT(user_input):
    messages.append({"role": "user", "content": user_input})
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = messages
    )
    ChatGPT_reply = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": ChatGPT_reply})
    return ChatGPT_reply

demo = gradio.Interface(fn=CustomChatGPT, inputs = "text", outputs = "text", title = "Software 1")

demo.launch(share=True)