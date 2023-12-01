"""Functions exchanging information from frontend with backend."""
from typing import Annotated, Optional

import requests
import streamlit as st
from constants import BACKEND_URL, USERNAME
from langchain.memory import ChatMessageHistory


def get_bots() -> list[dict]:
    """Get a list of available bots.

    Returns:
        list[dict]: a list of created bots.
    """
    bots = requests.get(f"{BACKEND_URL}/{USERNAME}/bots").json()

    return bots


def create_new_bot(bot_name: str) -> None:
    """Create a new bot with a given name.

    Args:
        bot_name (str): a name for a new bot
    """
    try:
        response = requests.put(
            f"{BACKEND_URL}/{USERNAME}/bots",
            json={"name": bot_name, "username": USERNAME},
        )
        assert response.status_code == 200
        st.success(f"Bot {bot_name} created")
    except Exception as e:
        st.warning(e)


def get_sources(bot_id: str) -> list[dict]:
    """Get a list of available source for the selected bot.

    Returns:
        list[dict]: a list of created sources for the bot.
    """
    sources = requests.get(f"{BACKEND_URL}/{USERNAME}/bots/{bot_id}/sources").json()

    return sources


def get_source(bot_id: str, source_id: str) -> dict:
    """Get a source with a given id.

    Args:
        bot_id (str): id of the bot to which the source belongs
        source_id (str): id of the source to get
    Returns:
        dict: a source with a given id.
    """
    source = requests.get(f"{BACKEND_URL}/{USERNAME}/bots/{bot_id}/sources/{source_id}")
    return source.json()


def get_source_file(bot_id: str, source_id: str) -> bytes:
    """Get a source file with a given id.

    Args:
        bot_id (str): id of the bot to which the source belongs
        source_id (str): id of the source to get
    Returns:
        bytes: a source file with a given id.
    """
    file = requests.get(
        f"{BACKEND_URL}/{USERNAME}/bots/{bot_id}/sources/{source_id}/file"
    ).content
    return file


def delete_source(bot_id: str, source_id: str) -> None:
    """Delete a source with a given id.

    Args:
        bot_id (str): id of the bot to which the source belongs
        source_id (str): id of the source to delete
    """
    try:
        response = requests.delete(
            f"{BACKEND_URL}/{USERNAME}/bots/{bot_id}/sources/{source_id}",
        )
        assert response.status_code == 200
        st.success(f"Source {source_id} deleted")
    except Exception:
        st.warning(response.text)


def create_new_source(
    source_name: str, source_type: str, bot_id: str, file: bytes = None, url: str = None
) -> None:
    """Create a new source for the bot with a given name.

    Args:
        source_name (str): a name for a new source for the bot
    """
    try:
        if source_type == "url":
            st.write("uploading url")
            response = requests.put(
                f"{BACKEND_URL}/{USERNAME}/bots/{bot_id}/sources",
                params={
                    "name": source_name,
                    "source_type": source_type,
                    "url": url,
                },
            )
        else:
            st.write("uploading file")
            response = requests.put(
                f"{BACKEND_URL}/{USERNAME}/bots/{bot_id}/sources",
                params={
                    "name": source_name,
                    "source_type": source_type,
                },
                files={"file": file},  # type: ignore
            )
        assert response.status_code == 200
        st.success(f"Source {source_name} added")
    except Exception:
        st.warning(f"Please provide a valid {source_type}")
        st.warning(response.text)


def create_new_prompt(prompt: str, bot_id: str) -> None:
    """Create a new prompt based on text from text area."""
    try:
        response = requests.put(
            f"{BACKEND_URL}/{USERNAME}/bots/{bot_id}/prompt", json={"prompt": prompt}
        )
        if response.status_code == 200:
            st.success("Prompt created", icon="👍")
        else:
            raise Exception(f"error: {response.status_code} {response.text}")
    except Exception as e:
        st.warning(e)


def get_prompt(bot_id: str) -> str:
    """Get the current prompt of a bot.

    Returns:
        str: bot's prompt.
    """
    prompt = requests.get(f"{BACKEND_URL}/{USERNAME}/bots/{bot_id}/prompt").json()[
        "prompt"
    ]

    return prompt


def get_conversations(bot_id: str) -> list[dict]:
    """Get a list of conversations involving given bot."""
    conversations = requests.get(
        f"{BACKEND_URL}/{USERNAME}/bots/{bot_id}/conversations"
    ).json()
    return conversations


def get_conversation(bot_id: str, conversation_id: str) -> dict | None:  # type: ignore
    """Get a conversation by id."""
    response = requests.get(
        f"{BACKEND_URL}/{USERNAME}/bots/{bot_id}/conversations/{conversation_id}"
    )
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 404:
        return None
    else:
        raise Exception(f"error: {response.status_code} {response.text}")


def get_bot(bot_id: str) -> dict:
    """Get a bot by id."""
    bot = requests.get(f"{BACKEND_URL}/{USERNAME}/bots/{bot_id}").json()
    return bot


def send_message(bot_id: str, conversation_id: str, message: str) -> tuple[str, str]:
    """Send a message to a bot and get a response."""
    response = requests.post(
        f"{BACKEND_URL}/{USERNAME}/bots/{bot_id}/chat",
        json={"message": message, "conversation_id": str(conversation_id)},
    ).json()
    return response["message"], response["conversation_id"]


def get_available_tools(bot_id: str) -> list[dict]:
    """Get a list of available tools for the selected bot.

    Returns:
        list[dict]: a list of created tools for the bot.
    """
    tools = requests.get(
        f"{BACKEND_URL}/{USERNAME}/bots/{bot_id}/tools/available_tools"
    ).json()

    return tools


def create_or_edit_tool(
    bot_id: str,
    tool_name: str,
    bot_description: str,
    user_variables: list,
    tool_id: Optional[str] = None,
) -> None:
    """Create a new tool for the bot with a given name and user variabes."""
    print(tool_id)

    response = requests.put(
        f"{BACKEND_URL}/{USERNAME}/bots/{bot_id}/tools",
        json={
            "id": tool_id,
            "name": tool_name,
            "bot_description": bot_description,
            "user_variables": user_variables,
        },
    )
    if response.status_code == 200:
        st.success(f"Tool {tool_name} added")
    else:
        st.danger(f"error: {response.status_code} {response.text}")


def get_tools(bot_id: str) -> list[dict]:
    """Get a list of available tools for the selected bot.

    Returns:
        list[dict]: a list of created tools for the bot.
    """
    tools = requests.get(f"{BACKEND_URL}/{USERNAME}/bots/{bot_id}/tools").json()

    return tools


def delete_tool(bot_id: str, tool_id: str) -> None:
    """Delete a tool with a given id."""
    requests.delete(f"{BACKEND_URL}/{USERNAME}/bots/{bot_id}/tools/{tool_id}")


def get_model(bot_id: str) -> str:
    """Get the current model of the bot."""
    model = requests.get(f"{BACKEND_URL}/{USERNAME}/bots/{bot_id}/model").json()
    return model


def get_available_models(bot_id: str) -> list[dict]:
    """Get a list of available models.

    Returns:
        list[dict]: a list of available models.
    """
    models = requests.get(
        f"{BACKEND_URL}/{USERNAME}/bots/{bot_id}/model/available_models"
    ).json()

    return models


def change_model(bot_id: str, model: dict) -> None:
    """Change the current model of the bot."""
    requests.put(
        f"{BACKEND_URL}/{USERNAME}/bots/{bot_id}/model",
        json=model,
    )
