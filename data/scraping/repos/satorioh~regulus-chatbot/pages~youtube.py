import streamlit as st
from config.global_config import (
    EMOJI,
    DISCLAIMER,
    SUMMARIZATION_MAX_SECONDS,
    SUMMARIZATION_TIME_LIMIT_HINT
)
from utils import format_time
from langchain.document_loaders import YoutubeLoader
from youtube_transcript_api import YouTubeTranscriptApi
from llm import get_youtube_summary

st.title(f"Regulus Youtube Summary")


def youtube_page():
    print("run youtube page...")
    hint_dom = st.markdown(DISCLAIMER)
    st.write("")

    def load_metadata(video_url):
        print(f"video url:{video_url}")
        loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=True,
                                                language=["en", "zh-Hans", "zh-Hant", "ja", "ko", "fr", "de", "ru",
                                                          "id",
                                                          "th"],
                                                translation="en")
        docs = loader.load()
        if len(docs) == 0:
            raise Exception("请确保当前视频包含字幕")
        print(docs)
        metadata = docs[0].metadata
        time = metadata['length']
        table = {
            'id': metadata['source'],
            '标题': metadata['title'],
            '描述': metadata['description'],
            '浏览量': metadata['view_count'],
            '发布日期': metadata['publish_date'],
            '时长': format_time(time),
            '作者': metadata['author']
        }
        st.table(table)
        st.image(metadata['thumbnail_url'], caption="缩略图")
        if time > SUMMARIZATION_MAX_SECONDS:
            st.warning(SUMMARIZATION_TIME_LIMIT_HINT, icon=EMOJI['warning'])
            return None
        else:
            return docs

    def load_summary(docs):
        if docs:
            summary = get_youtube_summary(docs)
            print(f"总结：{summary}")
            st.markdown(summary)

    def clear_text():
        st.session_state["url-text"] = ""

    with st.form("youtube-form", False):
        user_input = st.text_input('请输入 Youtube 链接：(e.g. https://www.youtube.com/watch?v=QsYGlZkevEg)',
                                   placeholder=SUMMARIZATION_TIME_LIMIT_HINT, key='url-text')
        url = user_input.strip()
        col1, col2 = st.columns([1, 1])
        with col1:
            btn_send = st.form_submit_button(
                "总结", use_container_width=True, type="primary")
        with col2:
            btn_clear = st.form_submit_button("清除", use_container_width=True, on_click=clear_text)

        if btn_send and url != "":
            docs = None
            try:
                with st.spinner('视频信息提取中...'):
                    docs = load_metadata(url)
            except Exception as e:
                print(f"视频提取失败:{e}")
                hint_dom.markdown(f"**视频提取失败：{e}**")
            if docs and len(docs) > 0:
                try:
                    with st.spinner('AI 总结中，时间可能较长，请耐心等待...'):
                        return load_summary(docs)
                except Exception as e:
                    print(f"生成总结失败:{e}")
                    hint_dom.markdown("**生成总结失败**")


youtube_page()
