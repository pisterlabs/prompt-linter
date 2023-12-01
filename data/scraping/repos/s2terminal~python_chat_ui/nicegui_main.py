import nicegui
import openai

def generate(e):
    input_text = e.sender.value
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": input_text}],
        max_tokens=100,
    )
    result.text = completion["choices"][0]["message"]["content"]

input_text = nicegui.ui.input().on('keydown.enter', generate)
result = nicegui.ui.label()

nicegui.ui.run()
