import os, nltk
from typing import Optional, Tuple
from scrap import scrap_url
import gradio as gr
from langchain.chains import ConversationChain
from threading import Lock
from langchain.document_loaders import TextLoader
from pdf_helper import convert_PDF
from youtube_utils import *
from load_chains import *


os.environ['OPENAI_API_KEY'] = 'your-openai-key-here'
url_pattern = re.compile(r'(([^\w\*])?(?:#|https|http|ftp|sftp|www)\S+)([^\w\*])?')


class ChatWrapper:

    def __init__(self):
        self.lock = Lock()
        self.chat_chain = load_chat_chain()
        self.qa_chain = None
        self.context_history = ""
        self.yt_history = ""

    def process_youtube_context(self, video_id, question):
        youtube_context = youtube_transcript(video_id=video_id)
        original_docs = get_youtube_chunks(youtube_context, chunk_size=200)
        youtube_chain = load_yt_chain(original_docs)
        lm_output = youtube_chain({"question": question, "history": self.yt_history})
        self.yt_history += "question: " + question + '\n'
        return lm_output

    def process_text_context(self, context, question):
        with open("url_content.txt", "w") as f:
            f.write(context)
        loader = TextLoader('url_content.txt')
        raw_documents = loader.load()
        self.qa_chain = load_context_qa_chain(raw_documents)
        lm_output = self.qa_chain({"question": question, "history": self.context_history})
        self.context_history += "question: " + question + '\n'
        return lm_output['answer']

    def __call__(
        self,  context: Optional[str], question: str, history: Optional[Tuple[str, str]], chain: Optional[ConversationChain]
    ):
        """Execute the chat functionality."""
        self.lock.acquire()
        try:
            history = history or []
            is_youtube = False
            # Run chain and append input. There is gonna be some bug
            if context:
                # context exists , create a hash id from the context and verify if it is new or not
                url_span = url_pattern.search(context)
                if url_span and context.startswith("https"):
                    if len(context) == url_span.endpos - url_span.pos:
                        if is_youtube_url(context):
                            is_youtube=True
                            video_id = get_youtube_id(context)
                            lm_output = self.process_youtube_context(video_id, question)
                        else:
                            context = scrap_url(context)
                if not is_youtube:
                    # create context hash id , check if hash id changed from the previous context , if yes then reload the chain
                    lm_output = self.process_text_context(context, question)

                if is_youtube:
                    if lm_output['sources']!='None' and lm_output['sources']!='':
                        out_video_url = get_output_video_url(video_id, lm_output)
                        text = "here"
                        embedded_text = f"<a href='{out_video_url}'>{text}</a>"
                        output = lm_output['answer']+'\n'+"you can find the relevant part of the video "+embedded_text
                    else:
                        output = lm_output['answer']
                else:
                    output = lm_output
            else:
                if not self.chat_chain:
                    self.chat_chain = load_chat_chain()
                output = self.chat_chain.run(input=question)
            # print("output" , output)
            history.append((question, output))
        except Exception as e:
            # print(e)
            raise e
        finally:
            self.lock.release()
        return history, history
    
chat = ChatWrapper()

block = gr.Blocks(css=".gradio-container {background-color: lightgray}")

with block:
    with gr.Row():
        gr.Markdown("<h2><center>ChatBot</center></h2>")
    
    with gr.Row():
        context = gr.Textbox(
                label="Context Paragraph",
                lines=7,
            )
        chatbot = gr.Chatbot()

    with gr.Row():
        message = gr.Textbox(
            label="What's your question?",
            placeholder="What's the answer to life, the universe, and everything?",
            lines=1,
        ).style(container=False)
        submit = gr.Button(value="Send", variant="secondary").style(full_width=False)

    gr.Examples(
        examples=[
            "Hi! How's it going?",
            "What should I do tonight?",
            "Whats 2 + 2?",
        ],
        inputs=message,
    )
    with gr.Row():
        uploaded_file = gr.File(
            label="Upload PDF file",
            file_count="single",
            type="file"
            ).style(full_width=False)
        with gr.Column():
            convert_button = gr.Button("Upload PDF!", variant="primary").style(full_width=False)
            out_placeholder = gr.HTML("<p><em>Output will appear in Context Paragraph:</em></p>")
    print("uploaded_file " , uploaded_file)
    convert_button.click(
            fn=convert_PDF,
            inputs=[uploaded_file],
            outputs=[context, out_placeholder],
        )
    # with gr.Column():
    #     gr.Markdown("Upload your own pdf here and ask about it. Files should be < 10MB to avoid upload issues - search for a PDF compressor online as needed. PDFs are truncated to 20 pages.")
    state = gr.State()
    agent_state = gr.State()
    submit.click(chat, inputs=[context, message, state, agent_state], outputs=[chatbot, state])
    message.submit(chat, inputs=[context, message, state, agent_state], outputs=[chatbot, state])


block.launch(debug=True, share=False, server_name='0.0.0.0')