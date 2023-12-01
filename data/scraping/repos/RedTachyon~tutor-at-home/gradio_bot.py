from anthropic import Anthropic
import gradio as gr
import time
import random
import json
import argparse

from dotenv import load_dotenv
from prompts import *
from tutor import Tutor

load_dotenv()
anthropic = Anthropic()




with open ('extracted/combinatorics/problem_base.json', 'r') as fil:
    problems = json.load(fil)

try:
    my_theme = gr.Theme.from_hub("ParityError/Anime")
except:
    my_theme = None


class Responser:
    def __init__(self):
        super().__init__()
        self.tutor = None
        self.message = None
    def reset(self, problem_name):
        self.tutor = Tutor(anthropic, problems[problem_name]['problem'], problems[problem_name]['solution'])
    def respond(self):
        resp = self.tutor.run(self.message)
        return resp

def run_demo(responser, args):
    allowed_paths = None
    css_code = None
    if args.unicorn:
        allowed_paths = ['/data/nikita/tutor-at-home/unicorns']
        css_code = ".gradio-container {background: url('file=unicorns/unicorn2.jpg')}"


    def new_statement(dropdown_value):
        #return gr.Textbox(problem_bank[dropdown_value]['problem'])
        return gr.Textbox(problems[dropdown_value]['problem']), gr.Textbox(value="", interactive=True), gr.Button(interactive=True)

    def generate_output(*args):
        output_text = "Hello!"
        return output_text

    def add_text(history, text):
        history += [(text, None)]
        responser.message = text
        return history, gr.Textbox(value="", interactive=True), gr.Button(interactive=True)

    def add_response(history):
        history += [(None, responser.respond())]
        return history, gr.Textbox(value="", interactive=True), gr.Button(interactive=True)

    
    def clear_chat():
        return [(None, "Please solve the problem displayed to your left and type in the solution.")], gr.Textbox(value="", interactive=True), gr.Button(interactive=True)

    def sleep():
        time.sleep(3)
        return gr.Textbox(interactive=True), gr.Button(interactive=True)

    with gr.Blocks(theme=my_theme, css=css_code) as demo:
        with gr.Row():

            # column for inputs
            with gr.Column():
                drop = gr.Dropdown(choices=list(problems.keys()), label = "Choose a problem")
                statement = gr.Textbox("",interactive=False, show_label=False)

            # column for outputs
            with gr.Column():
                chat = gr.Chatbot([(None, "Please choose a problem from the dropdown list.")], label="Chat")
                message = gr.Textbox("", interactive=False, label="Type your solution:")
                send_button = gr.Button("Send", interactive=False)

        drp_chng = drop.change(new_statement, drop, [statement, message, send_button])
        drp_chng = drp_chng.then(responser.reset, drop)
        drp_chng.then(clear_chat, outputs = [chat,message, send_button])
        btn_click = send_button.click(add_text, [chat, message], [chat,message, send_button])
        btn_click.then(add_response, [chat], [chat,message, send_button])
        txt_send = message.submit(add_text, [chat, message], [chat,message, send_button])
        txt_send.then(add_response, [chat], [chat,message, send_button])

    demo.launch(server_name="0.0.0.0", server_port=9011, share=args.share, allowed_paths=allowed_paths)
    return demo

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Arguments for gradio.")
    argparser.add_argument("--unicorn", action="store_true", help="Load unicorn")
    argparser.add_argument("--share", action="store_true", help="Share on public host")
    args = argparser.parse_args()
    run_demo(Responser(), args)
    #input("Press Enter to Shutdown...")
