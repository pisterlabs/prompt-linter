import os
import gradio as gr
from llama_index import VectorStoreIndex, SimpleDirectoryReader
import openai

# Method to set OpenAI API Key
def set_api(api):
  os.environ['OPENAI_API_KEY'] = f"{api}"
  # Uncomment below line if you are using it on colab
  #openai.api_key = f"{api}"
  return "API Set SuccessFul"

# Method to set Directory path
def set_path(path):
    try: 
        documents = SimpleDirectoryReader(path).load_data()
        global index
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist()
        return "Directory Path Set SuccessFul"
    except Exception as e:
        return f"{e}"

# Method that will generate response
def respond(message, chat_history):
    try:
        query_engine = index.as_query_engine()
        response = query_engine.query(f"{message}")
        response = str(response)
        chat_history.append((message, response))
        # time.sleep(2)
        return "", chat_history
    except Exception as e:
        response = e
        print(e)

# Code that is creating Web UI
with gr.Blocks() as demo:
  api = gr.Textbox(label="Your OpenAI API Key")
  api_btn = gr.Button(value="Submit API Key")
  api_btn.click(set_api, inputs=[api], outputs=[api])
  directory = gr.Textbox(label="Files Directory")
  path_btn = gr.Button(value="Set Path")
  path_btn.click(set_path, inputs=[directory], outputs=[directory])
  chatbot = gr.Chatbot()
  msg = gr.Textbox(label="Your Query")
  clear = gr.ClearButton([msg, chatbot])
  msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch(share=True)
