from tempfile import _TemporaryFileWrapper
import gradio as gr
import time

from tubegpt import TubeGPT
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
import os
import re
import whisper
import numpy as np

__tubegpt_dict = {}
__url_list = []
db_dir = "./db"


def initialize():
    if os.path.isdir(db_dir):
        subfolders = [(f.path, f.name) for f in os.scandir(db_dir) if f.is_dir()]
        if subfolders:
            print("found existing embedidings. loading them...")
            for folder in subfolders:
                url = f"https://www.youtube.com/watch?v={folder[1]}"
                chroma_folders = [f.path for f in os.scandir(folder[0]) if f.is_dir()]
                is_vision = False
                for channel in chroma_folders:
                    tubegpt = TubeGPT(url, db_dir)
                    if "video" in channel:
                        tubegpt.load_video(channel, embeddings=OpenAIEmbeddings())
                        is_vision = True
                    else:
                        tubegpt.load_audio(channel, embeddings=OpenAIEmbeddings())

                __tubegpt_dict[url] = {"tubegpt": tubegpt, "is_vision": is_vision}
                __url_list.append(url)

        print(__url_list)
        print(__tubegpt_dict)

    else:
        print("db dir not found, nothing to load")


def transcribe_audio(audio_file):
    model = whisper.load_model("small")
    result = model.transcribe(audio_file, fp16=False)
    return result["text"]


def refresh_url_list():
    return gr.Dropdown.update(choices=__url_list)


def query(url, question, query_options="audio"):
    if not url:
        return " url is empty"
    tubegpt = __tubegpt_dict[url]["tubegpt"]
    is_vision = __tubegpt_dict[url]["is_vision"]
    print(query_options)
    if query_options == "both":
        if not (is_vision):
            return "set query option to audio as no vision processing done"
        print("getting response using merged retreivers")
        response = tubegpt.query_merged(
            question=question, reader=ChatOpenAI(temperature=0)
        )
    elif query_options == "vision":
        if not (is_vision):
            return "set query option to audio as no vision processing done"
        print("getting response using only vision retreiver")
        response = tubegpt.query_vision(
            question=question, reader=ChatOpenAI(temperature=0)
        )
    else:
        print("getting response using only audio retreiver")
        response = tubegpt.query_audio(
            question=question, reader=ChatOpenAI(temperature=0)
        )

    return response


def process(
    url,
    is_vision=False,
    file_desc: _TemporaryFileWrapper = None,
    save_dir="/Users/bpakra200/Downloads/YouTubeclip ",
    progress=gr.Progress(),
):
    # MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
    # embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    # ---
    if is_vision:
        if not file_desc:
            return "upload description file", None
    progress(0, desc="starting processing")
    embeddings = OpenAIEmbeddings()
    tubegpt = TubeGPT([url], save_dir)
    progress(0.2, desc="processing audio")
    tubegpt.process_audio([url], save_dir, embeddings=embeddings)
    print("audio processing done")
    progress(0.5, desc="audio processing done")
    # __tubegpt_dict[url] = tubegpt
    if is_vision:
        progress(0.51, desc="processing vision data")
        time.sleep(0.1)
        tubegpt.process_vision_from_desc(
            file_path=file_desc.name, embeddings=embeddings
        )
        print("vision processing done")
        progress(0.8, desc="vision processing done, merging retreivers")
        time.sleep(0.5)
        tubegpt.merge_retreivers(
            [
                tubegpt.audio_db.db.as_retriever(search_kwargs={"k": 1}),
                tubegpt.vision_db.db.as_retriever(search_kwargs={"k": 1}),
            ]
        )
        print("retrievers merged")
        progress(0.9, desc="retrieivers merged")

    __tubegpt_dict[url] = {"tubegpt": tubegpt, "is_vision": is_vision}
    __url_list.append(url)
    print(__tubegpt_dict)
    print(__url_list)
    progress(1, desc="all processing done !")
    return ("all processing done", gr.Dropdown.update(choices=__url_list))


def read_textfile(file):
    with open(file, "r") as file_handle:
        file_text = file_handle.read()
    return file_text


def update_url(url):
    print("url is " + str(url))
    embed_html = extract_youtube_embed_code(url)
    return embed_html


def extract_youtube_embed_code(url):
    pattern = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com|youtu\.be)\/(?:watch\?v=|embed\/|v\/)?([a-zA-Z0-9_-]+)"
    match = re.match(pattern, url)
    if match:
        video_id = match.group(1)
        embed_code = f'<iframe width="560" height="315" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allowfullscreen></iframe>'
        return embed_code
    else:
        return None


def format_chat_prompt(message, chat_history):
    prompt = ""
    for turn in chat_history:
        user_message, bot_message = turn
        prompt = f"{prompt}\nUser: {user_message}\nAssistant: {bot_message}"
    prompt = f"{prompt}\nUser: {message}\nAssistant:"
    return prompt


def respond(url, message, query_options, chat_history):
    formatted_prompt = format_chat_prompt(message, chat_history)
    # bot_message = query(url,formatted_prompt,query_options)
    bot_message = query(url, message, query_options)
    # bot_message = client.generate(response,
    #                              max_new_tokens=1024,
    #                              stop_sequences=["\nUser:", "<|endoftext|>"]).generated_text
    chat_history.append((message, bot_message))
    print(f"chat_history: {chat_history}")
    return "", chat_history


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    with gr.Tab("Video Processing"):
        yt_url = gr.Textbox(label="VideoURL", value="Provide youtube URL here")
        is_vision = gr.Checkbox(label="vision", info="Process Vision Data?")
        # description_file = gr.File(label="Video Description File")
        file_text = gr.File(
            inputs="files", outputs="text", label="Upload Video Description file"
        )
        upload = gr.Button("process")
        text = gr.Markdown()

    with gr.Tab("Questions") as question:
        chatbot = gr.Chatbot(height=350)  # just to fit the notebook
        msg = gr.Textbox(label="Question for video?")
        url_processed = gr.Dropdown(
            choices=[],
            label="Processed url's",
            interactive=True,
            info="Select Youtube URL",
        )
        # embed_html = extract_youtube_embed_code(
        # url_processed
        # )
        embed_html = extract_youtube_embed_code("https://www.youtube.com/watch?v=_xASV0YmROc")
        html = gr.HTML(embed_html)
        state = gr.State(value="")
        with gr.Row():
            with gr.Column():
                audio = gr.Audio(source="microphone", type="filepath",streaming=True) 

        audio.stream(fn=transcribe_audio, inputs=[audio], outputs=[msg])
        query_options = gr.Dropdown(
            ["audio", "vision", "both"],
            info="Select Retreiver Model",
            label="Retreiver Model",
        )
        btn = gr.Button("Submit")
        refresh_url = gr.Button("Refresh URL List")
        clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")
        btn.click(
            respond,
            inputs=[url_processed, msg, query_options, chatbot],
            outputs=[msg, chatbot],
        )
        msg.submit(
            respond,
            inputs=[url_processed, msg, query_options, chatbot],
            outputs=[msg, chatbot],
        )  # Press enter to submit

    upload.click(
        process,
        inputs=[yt_url, is_vision, file_text],
        outputs=[text, url_processed],
        api_name="process",
    )  # command=on_submit_click
    initialize()
    refresh_url.click(refresh_url_list, inputs=[], outputs=url_processed)
    url_processed.input(fn=update_url, inputs=[url_processed] , outputs=html)

# demo.launch(share=True, server_port=7680)
if __name__ == "__main__":
    demo.queue(concurrency_count=20).launch(share=True, server_port=7680)
