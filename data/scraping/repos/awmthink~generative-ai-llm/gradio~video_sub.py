# %% import
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
import os

# %%

PROMPT = """
请将以下来自视频字幕的内容进行精确的翻译为中文。

###{text}###

翻译的内容可能涉及机器学习，深度学习，编程等领域，请保留这些领域的专有名词。
对于翻译后的内容，请适当的调整段落格式，以便好良好的可读性。
"""


def translate_text(text):
    chat_model = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo-16k")
    template = ChatPromptTemplate.from_template(PROMPT)
    chain = LLMChain(llm=chat_model, prompt=template)
    return chain.run(text=text)


# %%


def translate_all_subtitles(video_url):
    filename = video_url.split("/")[-1]
    save_dir = os.path.join("videos", filename)
    subtitles_loader = GenericLoader(
        YoutubeAudioLoader([video_url], save_dir), OpenAIWhisperParser()
    )
    subtitles = subtitles_loader.load()
    chunk_size = 4096
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, separators=["\n\n", "\n", ".", ""]
    )
    splitted_docs = splitter.split_documents(subtitles)
    translate_results = []
    for doc in splitted_docs:
        translate_results.append(translate_text(doc.page_content))
    return "\n\n".join(translate_results)


# %%
import gradio as gr


def main():
    with gr.Blocks() as demo:
        gr.Markdown("# 视频字幕翻译")
        with gr.Row():
            with gr.Column(scale=5):
                inputs = gr.Textbox(label="输入视频地址")
            with gr.Column(scale=1):
                btn = gr.Button(value="生成中文字幕")
        outputs = gr.Textbox(label="翻译结果", lines=10)
        btn.click(fn=translate_all_subtitles, inputs=[inputs], outputs=[outputs])
    demo.launch()


if __name__ == "__main__":
    main()
