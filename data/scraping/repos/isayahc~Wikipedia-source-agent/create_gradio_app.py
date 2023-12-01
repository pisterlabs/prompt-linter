import gradio as gr

from langchain import LLMChain
from langchain import PromptTemplate
from langchain.llms import Cohere

# from create_chain import chain as llm_chain
from create_chain import create_chain_from_template
from prompt import wikipedia_template, general_internet_template
from langchain.retrievers import CohereRagRetriever
from langchain.chat_models import ChatCohere

import os
from dotenv import load_dotenv



load_dotenv()  # take environment variables from .env. 
# https://pypi.org/project/python-dotenv/

COHERE_API_KEY = os.getenv("COHERE_API_KEY")


examples = [
    ["What is Cellular Automata and who created it?"],
    ["What is Cohere"],
    ["who is Katherine Johnson"],
]

def create_UI(llm_chain):
    with gr.Blocks() as demo:
    #     radio = gr.Radio(
    #     ["wikipedia only", "any website", "none"], label="What kind of essay would you like to write?", value="wikipedia only"
    # )
        radio = gr.Radio(
        ["wikipedia only", "any website", ], label="What kind of essay would you like to write?", value="wikipedia only"
    )

        
        chatbot = gr.Chatbot()
        msg = gr.Textbox(info="Enter your question here, press enter to submit query")
        clear = gr.Button("Clear")
        # submit_btn = gr.Button("Submit", variant="primary")

        gr.Examples(examples=examples, label="Examples", inputs=msg,)
        

        def user(user_message, history):
            return "", history + [[user_message, None]]

        def bot(history):
            print("Question: ", history[-1][0])
            bot_message = llm_chain.invoke(history[-1][0])

            bot_message = bot_message
            print("Response: ", bot_message)
            history[-1][1] = ""
            history[-1][1] += bot_message
            return history
        
        def change_textbox(choice):
            if choice == "wikipedia only":
                template = wikipedia_template
                llm_chain = create_chain_from_template(
                    template, 
                    rag, 
                    llm_model
                    )
                return llm_chain
            elif choice == "any website":
                template = general_internet_template
                llm_chain = create_chain_from_template(
                    template, 
                    rag, 
                    llm_model
                    )
                return llm_chain
            elif choice == "none":
                submit_btn = gr.Button("Submit", variant="primary")
                return gr.Textbox(lines=8, visible=True, value="Lorem ipsum dolor sit amet"), gr.Button("Submit", variant="primary")
            else:
                return gr.Textbox(visible=False), gr.Button(interactive=False)

        text = gr.Textbox(lines=2, interactive=True, show_copy_button=True)
        # radio.change(fn=change_textbox, inputs=radio, outputs=[text, submit_btn])
        radio.change(fn=change_textbox, inputs=radio, outputs=[text])
        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
        clear.click(lambda: None, None, chatbot, queue=False)
    return demo


if __name__ == "__main__":
    template = wikipedia_template
    prompt = PromptTemplate(template=template, input_variables=["query"])
    

    llm_model = ChatCohere(
        cohere_api_key=COHERE_API_KEY,
        )

    rag = CohereRagRetriever(llm=llm_model,)


    llm_chain = create_chain_from_template(
        template, 
        rag, 
        llm_model
        )

    demo = create_UI(llm_chain)
    demo.queue()
    # demo.launch()
    demo.launch(share=True)
    # pass