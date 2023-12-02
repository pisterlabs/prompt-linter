import json
from pathlib import Path
from openai import InvalidRequestError

import streamlit as st
from streamlit_card import card
from streamlit_extras.switch_page_button import switch_page
from langchain.callbacks import StreamlitCallbackHandler

from chatactor.model import Actor, CardModel
from chatactor.profiler import get_profiler
from chatactor.wikipedia import wikipedia2markdown
from functional.variables import COLS, DATADIR
from functional.utils import (
    download_images,
    on_click_card,
    divide_list,
    load_image,
)
from functional.component import settings
from functional.page import set_page_config, initial_session_state, description

set_page_config()
initial_session_state()

if st.session_state.openai_api_key is None or not st.session_state.openai_api_key:
    description()

    st.caption("Please return to the home page and enter your :red[OpenAI API key].")
else:
    with st.sidebar:
        settings()

    st.title("캐릭터를 선택해주세요 :seedling:")

    actors = [f"{DATADIR}/{file.stem}" for file in Path(DATADIR).glob("*.md")]

    for row in divide_list(actors, COLS):
        columns = st.columns(COLS)
        for file, col in zip(row, columns):
            with col, open(f"{file}.json", "r") as f:
                model = Actor(**json.load(f))
                model.image = f"{file}.jpg"
                model.content = f"""
                * {model.occupation}
                * {model.birth} ~ {model.death if model.death else '현재'}
                * {model.summary}
                
                """
                clicked = card(
                    title=model.name,
                    text=model.content,
                    image=load_image(f"{model.image}"),
                    styles={
                        "card": {
                            "width": "100%",
                            "height": "400px",
                            "margin": "0px",
                            "padding": "0px",
                        }
                    },
                    on_click=lambda: on_click_card(model),
                )

                if clicked:
                    switch_page("chat")

    st.divider()

    for message in st.session_state.profiler_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    agent = get_profiler(openai_api_key=st.session_state.openai_api_key)
    prompt = st.chat_input("Message", key="message")
    if prompt:
        st_callback = StreamlitCallbackHandler(
            st.container(),
            max_thought_containers=int(st.session_state.max_thought_containers),
            expand_new_thoughts=st.session_state.expand_new_thoughts,
            collapse_completed_thoughts=st.session_state.collapse_completed_thoughts,
        )
        try:
            output = agent.run(prompt, callbacks=[st_callback])
            summary = json.loads(output)  # json str -> dict
            model = Actor(**summary)  # dict -> Actor
            model.image = f"{DATADIR}/{model.name}.jpg"
            model.content = f"""
            * {model.occupation}
            * {model.birth} ~ {model.death if model.death else '현재'}
            * {model.summary}
            
            """

            with open(f"{DATADIR}/{model.name}.json", "w") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)  # save as json file

            with st.status(f"{model.name}를 조사하는 중.. :mag:", expanded=True):
                wikipedia2markdown(model.name)
            with st.status(f"{model.name}의 사진을 가져오는 중.. :camera:", expanded=True):
                download_images(model.name)

            output = summary

            if model.occupation is None:
                raise Exception("No occupation found.")

            if model.birth is None:
                raise Exception("No birth found.")

            st.session_state.model = model
            st.session_state.messages.clear()
            switch_page("chat")
        except InvalidRequestError:
            st.error(":warning: 조사에 실패했어요! 다시 시도해주세요. :warning:")

    else:
        st.caption(f"현재 {len(actors)}명의 캐릭터가 있어요. :mag:")
        st.caption("새로운 대화 상대를 검색해보세요. :hugging_face:")
        st.caption(" 세종대왕, 이순신, 핵 만든 오펜하이머, 원피스 루피 등등..")
