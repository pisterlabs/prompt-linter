from langchain.llms import OpenAI
import gradio as gr

llm = OpenAI(verbose=True, max_tokens=1024)

def open_ai_prompt(prompt_text):
    return llm(prompt_text)

demo = gr.Interface(
    fn=open_ai_prompt, 
    inputs="text", 
    outputs="text",
    allow_flagging=False)
    
demo.launch()