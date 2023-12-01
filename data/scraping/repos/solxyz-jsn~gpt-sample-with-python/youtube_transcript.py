# -*- coding: utf-8 -*-

"""Youtubeの字幕を取得する
"""

import requests
from io import BytesIO

from PIL import Image
from dotenv import load_dotenv
import streamlit as st

from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import YoutubeLoader
from langchain.prompts import PromptTemplate


# 環境変数の読み込み
load_dotenv()


def get_document(url):
    """
    Youtubeの字幕を取得する。

    Parameters
    ----------
    url : str
        YoutubeのURL。

    Returns
    -------
    Document
        Youtubeの字幕。

    """
    with st.spinner("Fetching Content ..."):
        loader = YoutubeLoader.from_youtube_url(
            url,
            add_video_info=True,  # タイトルや再生数も取得できる
            language=["en", "ja"],  # 英語→日本語の優先順位で字幕を取得
        )
        return loader.load()  # Document


def summarize(llm, docs, keyword):
    """
    Youtubeの字幕を要約する。

    Parameters
    ----------
    llm : ChatOpenAI
        ChatOpenAIのインスタンス。
    docs : Document
        Youtubeの字幕。
    keyword : str
        キーワード。
    ----------
    Returns
    -------
    str
        要約文。
    int
        OpenAI APIのコスト。

    """
    # プロンプトのテンプレート
    # ここでは、Youtubeの字幕を入力として、日本語の要約を出力する。
    # 指示語を英語にすることでトークンを節約する。
    prompt_template = """
                    Write a concise Japanese summary of the following transcript of Youtube Video. \
                    if keyword is given, summarize the part including the keyword.\

                    transcript:```{text}```\

                    日本語で必ず1000文字以内で簡潔にまとめること。
                    """
    # キーワードがあれば追加する
    if keyword:
        additional_prompt = f"""keyword: {keyword}"""
        prompt_template += additional_prompt
    # プロンプトの作成
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

    # 要約
    with get_openai_callback() as cb:
        chain = load_summarize_chain(
            llm,
            chain_type="stuff",
            verbose=True,
            prompt=PROMPT,
        )
        response = chain({"input_documents": docs}, return_only_outputs=True)
    return response["output_text"], cb.total_cost


def resize_image_from_url(image_url: str, new_width: int) -> Image:
    """
    URLから画像を取得し、指定された幅にリサイズする。

    Parameters
    ----------
    image_url : str
        画像のURL。
    new_width : int
        リサイズ後の画像の幅。

    Returns
    -------
    Image
        リサイズされた画像。

    """
    # イメージを取得
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))

    # リサイズ
    width, height = img.size
    new_height = int(height * new_width / width)
    img = img.resize((new_width, new_height))

    return img


st.title("Youtube ビデオの要約")
st.session_state.costs = []
url = st.text_input("Youtubeビデオ URL")
keyword = st.text_input("オプション：キーワード")
temperature_value = st.slider("Temprature：数値が低いほど精度が高く、高いほど曖昧になる", 0.0, 2.0, 1.0)

if st.button("要約開始"):
    document = get_document(url)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k")
    if document:
        with st.spinner("ChatGPT is typing ..."):
            output_text, cost = summarize(llm, document, keyword)
        st.session_state.costs.append(cost)
    else:
        output_text = None
    if output_text:
        st.subheader(document[0].metadata["title"])
        st.write(output_text)
        image = resize_image_from_url(document[0].metadata["thumbnail_url"], 600)
        st.image(image, caption="Thumbnail Image", use_column_width=True)
        st.write("Total Cost: ", sum(st.session_state.costs))
