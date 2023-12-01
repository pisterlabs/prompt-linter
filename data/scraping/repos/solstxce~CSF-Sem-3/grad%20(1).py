import gradio
import openai
# from gradio.components import inputs
from vars import KEY
openai.api_key = KEY
theme='JohnSmith9982/small_and_pretty'
def get_completion(Prompt):
    model="gpt-3.5-turbo"
    messages = [{"role": "user", "content": Prompt}]
    response = openai.ChatCompletion.create(model=model,messages=messages,temperature=0,)
    return response.choices[0].message["content"]
op=gradio.outputs.Textbox(label="API Response Text")
ip=gradio.inputs.Textbox(label="Prompt Text")
demo = gradio.Interface(fn=get_completion, inputs=ip, outputs=op,theme='JohnSmith9982/small_and_pretty')
    
demo.launch()  