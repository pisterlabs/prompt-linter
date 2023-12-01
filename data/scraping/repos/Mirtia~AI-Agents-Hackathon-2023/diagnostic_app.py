import base64
import json

import streamlit
from langchain.agents import (AgentExecutor,
                              create_csv_agent)
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.schema import messages_from_dict, messages_to_dict
from streamlit_chat import message

MODEL_NAME = "gpt-3.5-turbo-0613"
CSV_FILES = ["data/dataset.csv", "data/symptom_description.csv",
              "data/symptom_precaution.csv", "data/symptom_severity.csv"]


def generate_response(prompt: str, agent: AgentExecutor) -> str:
    return agent.run(prompt)


def get_prompt() -> str:
    return streamlit.text_input("You", placeholder="How are you feeling today?",
                                key="user_input", value=streamlit.session_state.user_input)


@streamlit.cache_resource
def initialize_agent(openai_api_key: str) -> AgentExecutor:
    return create_csv_agent(
        ChatOpenAI(
            temperature=0,
            model=MODEL_NAME,
            openai_api_key=openai_api_key),
        CSV_FILES,
        {"encoding": "utf-8", "on_bad_lines": "skip", "index_col": False},
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        memory=ConversationBufferMemory()
    )


@streamlit.cache_resource
def initialize_history() -> ChatMessageHistory:
    return ChatMessageHistory()


def import_history_from_file(history_file, history):
    if not history_file:
        return

    if not streamlit.session_state.get("history_imported", False):
        history_bytes = history_file.getvalue()
        history_messages = messages_from_dict(json.loads(history_bytes))
        for prompt, output in zip(history_messages[::2], history_messages[1::2]):
            streamlit.session_state.past.append(prompt.content)
            streamlit.session_state.generated.append(output.content)
            history.add_user_message(prompt.content)
            history.add_ai_message(output.content)

        streamlit.session_state["history_imported"] = True


def main():
    streamlit.title("Your General Practitioner 👩‍⚕️")
    openai_api_key = streamlit.sidebar.text_input("OpenAI API key")

    if not openai_api_key:
        streamlit.warning(
            "Please enter your OpenAI API key in the right field")
        return

    df_agent = initialize_agent(openai_api_key)

    history = ChatMessageHistory()

    streamlit.session_state.setdefault("generated", [])
    streamlit.session_state.setdefault("past", [])

    if "user_input" not in streamlit.session_state:
        streamlit.session_state.user_input = ""

    history_file = streamlit.file_uploader("Upload health history")
    import_history_from_file(history_file, history)

    prompt = get_prompt()

    if prompt and len(prompt) > 4:
        output = generate_response(prompt, agent=df_agent)
        streamlit.session_state.past.append(prompt)
        streamlit.session_state.generated.append(output)

        history.add_user_message(prompt)
        history.add_ai_message(output)

    if streamlit.session_state["generated"]:
        for i, (gen, past) in enumerate(zip(reversed(streamlit.session_state["generated"]),
                                            reversed(streamlit.session_state["past"]))):
            message(gen, key=str(i))
            message(past, is_user=True, key=str(i) + "_user")

    dicts = json.dumps(messages_to_dict(history.messages))
    download_icon = '<svg fill="currentColor" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="16" height="16"><path d="M0 0h16v16H0z" fill="none"/><path d="M9 8V3H7v5H4l4 4 4-4h-3zM4 13v1h8v-1H4z"/></svg>'
    href = f'{download_icon} <a href="data:file/history.json;base64,{base64.b64encode(dicts.encode()).decode()}" download="history.json">Download history</a>'
    streamlit.markdown(href, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
