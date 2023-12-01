import os
import gradio as gr
import pandas as pd
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI


os.environ["OPENAI_API_KEY"] = "V7d34eMGGxEmPuTXNgyRT3BlbkFJaGhyVNQeJ61V5pOQngns"
df = pd.read_csv('/content/Example_excel_sheet_test.csv')
df.head()

# Reads csv file and create agent to interact.
agent = create_csv_agent(OpenAI(temperature=0), '/content/Example_excel_sheet_test.csv',  verbose=True)


# Created by Rahul on 14-05-2023 chatbot function to take user input and provide fullfilment usinfg
def chatbot(input):
    if input:
        reply = agent.run(input) # CSV agent method called from langchain.agents
        return reply
    

# Created by Rahul on 14-05-2023 Gradio generated UI for user interaction
inputs = gr.inputs.Textbox(lines=7, label="Chat with AI")
print(inputs)
outputs = gr.outputs.Textbox(label="Reply")
print(outputs)
gr.Interface(fn=chatbot, inputs=inputs, outputs=outputs, title="AI Chatbot",
             description="Ask anything you want",
             theme="compact").launch(share=True)