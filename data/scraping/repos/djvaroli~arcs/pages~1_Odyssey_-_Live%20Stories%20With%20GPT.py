import base64
import time
from io import BytesIO
from typing import Generator, Literal

import pydantic
import streamlit as st
from openai import OpenAI
from PIL import Image
from rich import box
from rich.console import Console
from rich.table import Table


class ChatMessage(pydantic.BaseModel):
    role: str
    content: str | bytes
    content_type: Literal["text", "image", "audio"]

    def to_openai_dict(self) -> dict[str, str]:
        return {
            "role": self.role,
            "content": self.content,
        }

    def __eq__(self, __value: "ChatMessage") -> bool:
        return (
            self.role == __value.role
            and self.content == __value.content
            and self.content_type == __value.content_type
        )


class Messages(pydantic.BaseModel):
    messages: list[ChatMessage]

    def append(self, message: ChatMessage) -> None:
        self.messages.append(message)

    @property
    def text_messages(self) -> list[ChatMessage]:
        return [message for message in self.messages if message.content_type == "text"]

    def role_ordered_messages(self) -> list[list[ChatMessage]]:
        ordered_messages_by_role: list[list[ChatMessage]] = []
        prev_role = None
        for message in self.messages:
            if message.role != prev_role:
                ordered_messages_by_role.append([])
            ordered_messages_by_role[-1].append(message)
            prev_role = message.role

        return ordered_messages_by_role

    def __contains__(self, message: ChatMessage) -> bool:
        return message in self.messages

    def __iter__(self) -> Generator[ChatMessage, None, None]:
        return iter(self.messages)

    def __len__(self) -> int:
        return len(self.messages)


class ChatSettings(pydantic.BaseModel):
    narrator_model: str
    narrator_temperature: float
    tts_model: str


def raise_if_not_valid_api_key(
    client: OpenAI,
) -> None:
    client.completions.create(model="davinci", prompt="This is a test.", max_tokens=5)


def generate_image(
    client: OpenAI,
    prompt: str,
) -> Image.Image:
    """Generates an image using the OpenAI API.

    Args:
        client (OpenAI): _description_
        prompt (str): _description_

    Returns:
        Image.Image:
    """
    resp = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        n=1,
        size="1024x1024",
        response_format="b64_json",
    )

    image_b64_string = resp.data[0].b64_json

    return Image.open(BytesIO(base64.b64decode(image_b64_string)))


def generate_text(
    client: OpenAI,
    model: str,
    messages: Messages,
    temperature: float = 1.0,
) -> str:
    text_messages = messages.text_messages
    resp = client.chat.completions.create(
        model=model,
        messages=[message.to_openai_dict() for message in text_messages],
        temperature=temperature,
    )

    return resp.choices[0].message.content


def timed_popup(
    message: str,
    kind: Literal["info", "error", "warning", "success"],
    timeout: int = 3,
) -> None:
    """Displays a popup message for a specified amount of time.

    Args:
        message (str): The message to display.
        kind (Literal["info", "error", "warning", "success"]): The type of message.
        timeout (int, optional): The amount of time to display the message. Defaults to 3.
    """
    if kind == "info":
        popup = st.info(message)
    elif kind == "error":
        popup = st.error(message)
    elif kind == "warning":
        popup = st.warning(message)
    elif kind == "success":
        popup = st.success(message)

    time.sleep(timeout)
    popup.empty()


def append_message(
    role: str, content: str, content_type: str, allow_duplicates: bool = False
) -> None:
    message = ChatMessage(role=role, content=content, content_type=content_type)
    session_messages: Messages = st.session_state.messages
    if allow_duplicates or message not in session_messages:
        session_messages.append(message)


def print_box(
    content: str,
    style: str = "bold white on black",
) -> None:
    # print a table with a single column and row making it look like a box
    console = Console()
    table = Table(show_header=False, box=box.DOUBLE_EDGE)
    table.add_column()
    table.add_row(str(content))
    console.print(table, style=style)


def hide_api_key_components_callback() -> None:
    if "hide_api_key_componets" not in st.session_state:
        st.session_state.hide_api_key_componets = True


def is_chat_started() -> bool:
    return st.session_state.get("chat_started", False)


def is_api_client_set() -> bool:
    return "client" in st.session_state


def start_chat() -> None:
    st.session_state.chat_started = True


def set_chat_settings(
    narrator_model: str, narrator_temperature: float, tts_model: str
) -> None:
    st.session_state.chat_settings = ChatSettings(
        narrator_model=narrator_model,
        narrator_temperature=narrator_temperature,
        tts_model=tts_model,
    )


def get_chat_settings() -> ChatSettings:
    return st.session_state["chat_settings"]


IMAGE_WIDTH = 375
st.title("Odyssey - Live Storytelling")
st.write(
    "Odyssey is an interactive storytelling experience that allows you to create stories with the help of AI.\
Set the stage with your first message, and let GPT continue and narrate the story,\n\
and DALL-E to generate a cool illustration to go along with it. Then it's your turn to continue the story. You can go on as long as you like (just remember to keep tabs on spending)!"
)

st.sidebar.info(
    "The OpenAI API Key is stored in Streamlit's session state, and is not saved on disk. \
    The session state will reset if you reload the page. \
    For maximum security, please create a dedicated OpenAI API key and set appropriate spending limits."
)
st.sidebar.warning(
    "GPT4 usage can become expensive quickly. Please ensure to set spending limits in your OpenAI API dashboard."
)


st.info(
    "If you do not have an API key, please visit https://platform.openai.com/api-keys to create one. \
    Please ensure to set spending limits in your OpenAI API dashboard at https://platform.openai.com/usage"
)
open_ai_api_key_input = st.text_input(
    "OpenAI API Key (starts with 'sk-')",
    type="password",
    key="open-ai-api-key-input",
    disabled=is_api_client_set(),
    help="To get an API key, visit https://platform.openai.com/api-keys",
)


if st.button("Save", key="save-api-key-button", disabled=is_api_client_set()):
    try:
        client = OpenAI(api_key=open_ai_api_key_input)
        raise_if_not_valid_api_key(client)
        st.session_state.client = client
        timed_popup("API Key Validated and Set!", "success", timeout=3)
        st.rerun()

    except Exception as e:
        st.error(e)
        st.stop()

if "client" not in st.session_state:
    st.stop()

if not is_chat_started():
    narrator_model = st.selectbox(
        "Narrator Model",
        ["gpt-4-1106-preview", "gpt-3.5-turbo"],
        help="GPT3.5 is cheaper, but GPT4 is more creative.",
    )
    tts_model = st.selectbox(
        "Text-to-Speech Model",
        ["tts-1", "tts-1-hd"],
        help="tts-1 is faster, but tts-1-hd is higher quality.",
    )
    narrator_temperature = st.number_input(
        "Narrator Temperature",
        min_value=0.0,
        value=1.05,
        step=0.05,
        help="Higher temperature results in more creative responses, lower temperature in more predictable responses.",
    )

    set_chat_settings(
        narrator_model=narrator_model,
        narrator_temperature=narrator_temperature,
        tts_model=tts_model,
    )

else:
    st.write("Narrator Model: ", get_chat_settings().narrator_model)
    st.write("Narrator Temperature: ", get_chat_settings().narrator_temperature)
    st.write("Text-to-Speech Model: ", get_chat_settings().tts_model)


if "messages" not in st.session_state:
    st.session_state.messages = Messages(messages=[])

st.markdown("### Interactive Story")
for role_ordered_message in st.session_state.messages.role_ordered_messages():
    role = role_ordered_message[0].role

    # do not show system message
    if role == "system":
        continue

    with st.chat_message(role):
        for message in role_ordered_message:
            # only show user text messages
            if message.content_type == "text" and message.role == "user":
                st.write(message.content)

            elif message.content_type == "image":
                img = Image.open(BytesIO(message.content))
                st.image(img, width=IMAGE_WIDTH)

            elif message.content_type == "audio":
                st.audio(message.content)


narrator_system_prompt = """
You are an expert narrator and storyteller. 
You will pair with the user (reader) to create the story together. The user (reader) will provide the first prompt to start the story.
Make use of literary techniques such as foreshadowing, suspense, cliffhangers, and plot twists when appropriate.
Ensure that generated text ends in a way that allows the reader to continue the story. Limit your responses to a maximum of 8 - 10 sentences.
**DO NOT ADDRESS THE USER (READER) DIRECTLY.**
**DO NOT MENTION THE USER (READER) IN THE STORY**
**ENSURE THAT YOUR RESPONSES ADHERE TO ETHICAL AND MORAL CONSIDERATIONS.**
"""

append_message(
    role="system",
    content=narrator_system_prompt,
    content_type="text",
    allow_duplicates=False,
)

if prompt := st.chat_input("Your turn to continue the story..."):
    if not is_chat_started():
        start_chat()

    append_message(role="user", content=prompt, content_type="text")

    with st.chat_message("user"):
        st.write(prompt)

    client: OpenAI = st.session_state.client
    with st.chat_message("assistant"):
        with st.spinner("Continuing story..."):
            story_continuation = generate_text(
                client,
                model=get_chat_settings().narrator_model,
                messages=st.session_state.messages,
            )

        with st.spinner("Generating illustrations..."):
            illustration = generate_image(client, prompt=story_continuation)

        with st.spinner("Generating narration..."):
            continuation_narration = client.audio.speech.create(
                model=get_chat_settings().tts_model,
                voice="echo",
                input=story_continuation,
            )

        append_message(
            role="assistant", content=story_continuation, content_type="text"
        )

        with BytesIO() as output:
            illustration.save(output, format="PNG")
            illustration_bytes = output.getvalue()

        append_message(
            role="assistant", content=illustration_bytes, content_type="image"
        )
        append_message(
            role="assistant",
            content=continuation_narration.read(),
            content_type="audio",
        )

        st.image(illustration, width=IMAGE_WIDTH)
        st.audio(continuation_narration.read())
