from typing import List, Tuple

from fastapi import FastAPI
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from nicegui import Client, app, ui

OPENAI_API_KEY = "not-set"  # TODO: set your OpenAI API key here

llm = ConversationChain(llm=ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY))

messages: List[Tuple[str, str, str]] = []
thinking: bool = False


def init(server: FastAPI) -> None:
    @ui.refreshable
    async def chat_messages() -> None:
        for name, text in messages:
            ui.chat_message(text=text, name=name, sent=name == "You", avatar="https://i.pravatar.cc/300?u=1")
        if thinking:
            ui.spinner(size="3rem").classes("self-center")
        await ui.run_javascript("window.scrollTo(0, document.body.scrollHeight)", respond=False)

    @ui.page("/")
    async def main(client: Client):
        async def send() -> None:
            global thinking
            message = text.value
            messages.append(("You", text.value))
            thinking = True
            text.value = ""
            chat_messages.refresh()

            try:
                response = await llm.arun(message)
            except Exception as e:
                response = "Sorry, I don't understand. {e}"
            messages.append(("Bot", response))
            thinking = False
            chat_messages.refresh()

        anchor_style = r"a:link, a:visited {color: inherit !important; text-decoration: none; font-weight: 500}"
        ui.add_head_html(f"<style>{anchor_style}</style>")
        await client.connected()

        with ui.column().classes("w-full max-w-2xl mx-auto items-stretch"):
            await chat_messages()

        with ui.footer().classes("bg-white"), ui.column().classes("w-full max-w-3xl mx-auto my-6"):
            with ui.row().classes("w-full no-wrap items-center"):
                placeholder = (
                    "message"
                    if OPENAI_API_KEY != "not-set"
                    else "Please provide your OPENAI key in the Python script first!"
                )
                text = (
                    ui.input(placeholder=placeholder)
                    .props("rounded outlined input-class=mx-3")
                    .classes("w-full self-center")
                    .on("keydown.enter", send)
                )

    ui.run_with(
        server,
        title="Chatbot",
        storage_secret="pick your private secret here",  # NOTE setting a secret is optional but allows for persistent storage per user
    )
