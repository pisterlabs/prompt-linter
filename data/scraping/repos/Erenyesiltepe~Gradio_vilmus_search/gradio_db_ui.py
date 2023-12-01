import gradio as gr
import query_milvus as qm
import openai_query as oq

def respond(options ,message, chat_history):
    if options==None or message.replace(" ","")=="":
        chat_history.append((message, "Please select a model and don't leave text field empty"))
    else:
        if options=="cohere":
            response= qm.searchDB(message)
            print("cohere")
        elif options=="openai":
            response= oq.searchDB(message)
            print("openai")
        else:
            print("something is wrong")
        chat_history.append((message, response))

    return "", chat_history

with gr.Blocks() as demo:
    options=gr.Radio(["cohere", "openai"], label="Models")
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    msg.submit(fn=respond, inputs=[options,msg, chatbot], outputs=[msg, chatbot])

if __name__ == "__main__":
    demo.launch()