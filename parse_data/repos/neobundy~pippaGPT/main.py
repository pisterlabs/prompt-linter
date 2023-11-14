# Standard library imports
from datetime import datetime
import os
import re
import subprocess
from uuid import uuid4
from dotenv import load_dotenv

# Third-party library imports
import openai
import requests
import streamlit as st
from langchain.callbacks import get_openai_callback
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationTokenBufferMemory,
    ConversationSummaryMemory,
    ConversationSummaryBufferMemory,
    ZepMemory,
)
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from pathlib import Path

# Local application/library specific imports
import helper_module
from vectordb import retrieval_qa_run, display_vectordb_info
import token_counter
import tts
import characters
import agents
import settings


class StreamHandler(BaseCallbackHandler):
    def __init__(
        self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""
    ):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


def get_zep_session_id():
    if "zep_session_id" in st.session_state:
        return st.session_state.zep_session_id
    else:
        return str(uuid4())


def init_page():
    st.set_page_config(page_title=settings.PROJECT_TITLE, page_icon="🤗")
    st.header(f"{settings.PROJECT_TITLE}", anchor="top")

    st.image(settings.AVATAR_AI)

    st.markdown(""" [Go to Bottom](#bottom) """, unsafe_allow_html=True)
    st.sidebar.caption(f"v{settings.VERSION}")
    st.sidebar.subheader("Options")


def init_session_state():
    st.session_state.setdefault("transcript", "")
    st.session_state.setdefault("costs", [])
    st.session_state.setdefault("streaming_enabled", settings.STREAMING_ENABLED)
    st.session_state.setdefault(
        "custom_instructions", get_custom_instructions(default=True)
    )
    st.session_state.setdefault("conversation_history", [])
    st.session_state.setdefault("zep_session_id", str(uuid4()))
    st.session_state.setdefault("memory_type", settings.DEFAULT_MEMORY_TYPE)


def get_dynamic_token_parameters(model_name, memory):
    custom_instructions = get_custom_instructions()
    ci_tokens = token_counter.token_counter(custom_instructions)
    history_tokens = token_counter.token_counter(
        convert_messages_to_str(get_messages_from_memory(memory))
    )
    user_input_tokens = settings.AVERAGE_USER_INPUT_TOKENS

    input_tokens = ci_tokens + history_tokens + user_input_tokens

    max_token_hard_cap = settings.MAX_TOKENS_4K  # Default value

    if "gpt-4" in model_name:
        max_token_hard_cap = (
            settings.MAX_TOKENS_16K if "16k" in model_name else settings.MAX_TOKENS_8K
        )
    elif "gpt-3.5" in model_name and "16k" in model_name:
        max_token_hard_cap = settings.MAX_TOKENS_16K

    comp_tokens_margin = max_token_hard_cap - input_tokens
    max_tokens = comp_tokens_margin - settings.TOKEN_SAFE_MARGIN

    params = {
        "ci_tokens": ci_tokens,
        "history_tokens": history_tokens,
        "user_input_tokens": user_input_tokens,
        "input_tokens": input_tokens,
        "comp_tokens_margin": comp_tokens_margin,
        "max_tokens": max_tokens,
        "max_token_hard_cap": max_token_hard_cap,
    }

    if max_tokens < 0:
        helper_module.log("params", "error")
        raise Exception(
            f"Model unusable. Requires more completion tokens: {max_tokens}"
        )

    return params


def update_context_window(context_window):
    custom_instructions = get_custom_instructions()

    # 1. Sliding Window: ConversationBufferWindowMemory - retains a specified number of messages.
    # 2. Token Buffer: ConversationTokenBufferMemory - retains messages based on a given number of tokens.
    # 3. Summary Buffer: ConversationSummaryBufferMemory - retains a summarized history while also storing all messages.
    # 4. Summary: ConversationSummaryMemory - retains only the summary.
    # 5. Buffer: ConversationBufferMemory - the most basic memory type that stores the entire history of messages as they are.
    # 6. Zep: vector store

    memory_panel = st.sidebar.expander("Memory Types")
    with memory_panel:
        memory_type = st.radio(
            "✏️",
            settings.MEMORY_TYPES,
            index=settings.MEMORY_TYPES.index(settings.DEFAULT_MEMORY_TYPE),
        )

        # Helper GPT Model for chat history summary
        # max_tokens: Remember that it's effectively available completion tokens excluding all input tokens(ci + user input) from hard cap

        token_parameters = get_dynamic_token_parameters(
            settings.DEFAULT_GPT_HELPER_MODEL, context_window
        )
        max_tokens = token_parameters["max_tokens"]

        llm = ChatOpenAI(
            temperature=settings.DEFAULT_GPT_HELPER_MODEL_TEMPERATURE,
            model_name=settings.DEFAULT_GPT_HELPER_MODEL,
            max_tokens=max_tokens,
        )

        # memory_key: the variable name in the prompt template where context window goes in
        # You are a chatbot having a conversation with a human.
        #
        # {context_window}
        # Human: {human_input}
        # Chatbot:

        if memory_type.lower() == "sliding window":
            updated_context_window = ConversationBufferWindowMemory(
                llm=llm,
                memory_key="context_window",
                chat_memory=context_window,
                input_key="human_input",
                k=settings.SLIDING_CONTEXT_WINDOW,
                return_messages=True,
            )
        elif memory_type.lower() == "token":
            updated_context_window = ConversationTokenBufferMemory(
                llm=llm,
                memory_key="context_window",
                chat_memory=context_window,
                input_key="human_input",
                max_token_limit=settings.MAX_TOKEN_LIMIT_FOR_SUMMARY,
                return_messages=True,
            )
        elif memory_type.lower() == "summary":
            updated_context_window = ConversationSummaryMemory.from_messages(
                llm=llm,
                memory_key="context_window",
                chat_memory=context_window,
                input_key="human_input",
                return_messages=True,
            )
            st.caption(
                updated_context_window.predict_new_summary(
                    get_messages_from_memory(updated_context_window), ""
                )
            )
        elif memory_type.lower() == "summary buffer":
            updated_context_window = ConversationSummaryBufferMemory(
                llm=llm,
                memory_key="context_window",
                chat_memory=context_window,
                input_key="human_input",
                max_token_limit=settings.MAX_TOKEN_LIMIT_FOR_SUMMARY,
                return_messages=True,
            )
            st.caption(
                updated_context_window.predict_new_summary(
                    get_messages_from_memory(updated_context_window), ""
                )
            )
        elif memory_type.lower() == "zep":
            # Zep uses a vector store and is not compatible with other memory types in terms of the context window.
            # When you change the memory type to an incompatible one, simply load the latest snapshot.
            zep_session_id = get_zep_session_id()
            updated_context_window = ZepMemory(
                session_id=zep_session_id,
                url=settings.ZEP_API_URL,
                memory_key="context_window",
                input_key="human_input",
                return_messages=True,
            )
            zep_summary = updated_context_window.chat_memory.zep_summary
            if zep_summary:
                st.caption(zep_summary)
            else:
                st.caption("Summarizing...please be patient.")
            if settings.DEBUG_MODE:
                helper_module.log(
                    f"Zep Summary - {updated_context_window.chat_memory.zep_summary}", "debug"
                )
        else:
            updated_context_window = ConversationBufferMemory(
                llm=llm,
                memory_key="context_window",
                chat_memory=context_window,
                input_key="human_input",
                return_messages=True,
            )

    view_custom_instructions = st.expander("View Custom Instructions")
    with view_custom_instructions:
        if st.button("Load default custom instructions"):
            set_custom_instructions(get_custom_instructions(default=True))
        new_custom_instructions = st.text_area(label="Change Custom Instructions:",
                                               value=get_custom_instructions(),
                                               height=200,
                                               max_chars=settings.MAX_NUM_CHARS_FOR_CUSTOM_INSTRUCTIONS,
                                               )
        if new_custom_instructions != custom_instructions:
            set_custom_instructions(new_custom_instructions)
            st.success(f"✏️ Custom instructions updated.")
            custom_instructions = new_custom_instructions
        handle_message(SystemMessage(content=custom_instructions), 0)

    st.session_state.memory_type = memory_type
    return updated_context_window


def get_role_from_message(message):
    if isinstance(message, AIMessage):
        return settings.VOICE_NAME_AI.title()
    elif isinstance(message, HumanMessage):
        return settings.VOICE_NAME_HUMAN.title()
    elif isinstance(message, SystemMessage):
        return settings.VOICE_NAME_SYSTEM.title()
    else:
        return "Unknown"


def get_messages_from_memory(memory_type):
    if isinstance(memory_type, StreamlitChatMessageHistory):
        # st.sessions_state.context_window - context window variable
        return memory_type.messages
    else:
        # one of LangChain's memory types
        return memory_type.chat_memory.messages


def convert_messages_to_str(messages):
    messages = [m.content for m in messages]
    messages_str = " ".join(messages)

    return messages_str


def save_user_input_to_file(user_input):
    helper_module.save_to_text_file(user_input, settings.USER_INPUT_SAVE_FILE)


def select_model(memory):
    st.session_state.streaming_enabled = st.sidebar.checkbox(
        "Enable Streaming", value=settings.STREAMING_ENABLED, key="streaming"
    )
    model_names = helper_module.get_openai_model_names(gpt_only=True)
    model = st.sidebar.selectbox(
        "Choose a model:",
        model_names,
        index=model_names.index(settings.DEFAULT_GPT_MODEL),
    )
    st.session_state.model_name = model
    parameters_options = st.sidebar.expander("Parameters")
    with parameters_options:
        temperature = st.slider(
            "Temperature:",
            min_value=0.0,
            max_value=2.0,
            value=settings.DEFAULT_GPT_MODEL_TEMPERATURE,
            step=0.01,
        )

    token_parameters = get_dynamic_token_parameters(st.session_state.model_name, memory)
    ci_tokens = token_parameters["ci_tokens"]
    history_tokens = token_parameters["history_tokens"]
    user_input_tokens = token_parameters["user_input_tokens"]
    input_tokens = token_parameters["input_tokens"]
    comp_tokens_margin = token_parameters["comp_tokens_margin"]
    max_tokens = token_parameters["max_tokens"]
    max_token_hard_cap = token_parameters["max_token_hard_cap"]

    with parameters_options:
        max_completion_tokens = st.slider(
            "Max Completion Tokens:",
            min_value=settings.DEFAULT_MIN_COMPLETION_TOKENS,
            max_value=min(comp_tokens_margin, max_tokens),
            value=settings.DEFAULT_COMPLETION_TOKENS,
            step=10,
        )

        st.caption(f"Model Name: {st.session_state.model_name}")
        st.caption(f"Custom Instructions Tokens: {ci_tokens}")
        st.caption(f"Chat History Tokens: {history_tokens}")
        st.caption(f"User Input Tokens: {user_input_tokens}")
        st.caption(
            f"Input Tokens(ci_tokens + history_tokens + user_input_tokens): {input_tokens}"
        )
        st.caption(f"Max Token Hard Cap: {max_token_hard_cap}")
        st.caption(
            f"Completion Token Margin(max_token_hard_cap - input_tokens): {comp_tokens_margin}"
        )
        st.caption(
            f"Max Tokens Set(comp_tokens_margin - settings.TOKEN_SAFE_MARGIN({settings.TOKEN_SAFE_MARGIN})): {max_tokens}"
        )

    prompt_template = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template("{system_input}"),
            # The `variable_name` here is what must align with memory
            # context_window is a list of messages: [HumanMessage(), AIMessage(),]
            MessagesPlaceholder(variable_name="context_window"),
            HumanMessagePromptTemplate.from_template("{human_input}"),
        ]
    )

    prompt_template.format(
        system_input=get_custom_instructions(),
        context_window=get_messages_from_memory(memory),
        human_input="",
    )

    return LLMChain(
        llm=ChatOpenAI(
            streaming=st.session_state.streaming_enabled,
            temperature=temperature,
            model_name=model,
            max_tokens=max_completion_tokens,
        ),
        prompt=prompt_template,
        memory=memory,
        verbose=settings.LLM_VERBOSE,
    )


def set_custom_instructions(ci=""):
    st.session_state.custom_instructions = ci
    helper_module.log(f"Custom Instructions Set: {ci}", "info")


def get_custom_instructions(default=False):
    if default:
        return characters.CUSTOM_INSTRUCTIONS

    return st.session_state.custom_instructions


# TO-DO: just a placeholder for future use
def format_display_text(text: str):
    # Latex expressions are handled via system message:
    #   If you need to use Latex in your response, please use the following format: $ [latex expression] $
    # In Streamlit st.markdown, Latex expressions are handled via $ [latex expression] $ (inline), or $$ [latex expression] $$ (block)

    return text


def handle_message(message, i):
    if isinstance(message, AIMessage):
        with st.chat_message("assistant", avatar=settings.AVATAR_AI):
            st.markdown(format_display_text(message.content))
            button_name = "_st_read_pippa_button_" + str(i)
            if st.button("Speak", key=button_name):
                with st.spinner("Pippa is speaking ..."):
                    tts.generate_audio(
                        message.content, character=settings.VOICE_NAME_AI
                    )
                    tts.play_audio(settings.VOICE_FILE_AI)
    elif isinstance(message, HumanMessage):
        with st.chat_message("user", avatar=settings.AVATAR_HUMAN):
            st.markdown(format_display_text(message.content))
            button_name = "_st_read_bundy_button_" + str(i)
            if st.button("Speak", key=button_name):
                with st.spinner("Bundy is speaking ..."):
                    tts.generate_audio(
                        message.content, character=settings.VOICE_NAME_HUMAN
                    )
                    tts.play_audio(settings.VOICE_FILE_HUMAN)
    elif isinstance(message, SystemMessage):
        with st.chat_message("system", avatar=settings.AVATAR_SYSTEM):
            st.markdown(format_display_text(message.content))
            button_name = "_st_read_system_button_" + str(i)
            if st.button("Speak", key=button_name):
                with st.spinner("System is speaking ..."):
                    tts.generate_audio(
                        message.content, character=settings.VOICE_NAME_SYSTEM
                    )
                    tts.play_audio(settings.VOICE_FILE_SYSTEM)
    else:
        st.write(f"System message: {message.content}")


def display_tts_panel():
    tts_options = st.sidebar.expander("TTS/STT")
    with tts_options:
        st.session_state.use_audio = st.checkbox("Use Audio", value=False)
        if st.button("Record"):
            r = requests.get(f"{settings.AUDIO_SERVER_URL}/start_recording")
            if r.status_code == 200:
                st.success("Start recording...")
            else:
                st.error("Couldn't start recording.")
        if st.button("Stop"):
            r = requests.get(f"{settings.AUDIO_SERVER_URL}/stop_recording")
            if r.status_code == 200:
                st.success("Successfully recorded.")
                audio_file = open(settings.TRANSCRIPTION_TEMP_AUDIO_FILE, "rb")
                st.session_state.transcript = openai.Audio.transcribe(
                    settings.OPENAI_AUDIO_MODEL, audio_file
                )["text"]
            else:
                st.error("Couldn't stop recording.")


def append_to_full_conversation_history(message):
    st.session_state.conversation_history.append(message)


def get_full_conversation_history():
    return st.session_state.conversation_history


def reset_full_conversation_history():
    st.session_state.conversation_history = []


def export_conversation(snapshot=False):
    json_message = [
        {"role": get_role_from_message(m), "content": m.content}
        for m in get_full_conversation_history()
    ]

    if not os.path.exists(settings.CONVERSATION_SAVE_FOLDER):
        os.makedirs(settings.CONVERSATION_SAVE_FOLDER)

    if snapshot:
        filepath = f"{settings.CONVERSATION_SAVE_FOLDER}/{settings.SNAPSHOT_FILENAME}"
    else:
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        save_folder = Path(settings.CONVERSATION_SAVE_FOLDER)
        save_folder.mkdir(parents=True, exist_ok=True)
        filepath = f"{settings.CONVERSATION_SAVE_FOLDER}/conversation-{now}.json"
    helper_module.save_conversation_to_json(json_message, filepath)
    return filepath


def load_conversation(memory, conversation):
    if Path(conversation).exists():
        loaded_messages = helper_module.load_conversation_from_json(conversation)
        last_num = len(get_full_conversation_history()) + 10
        reset_full_conversation_history()
        memory.chat_memory.messages.clear()
        for message in loaded_messages:
            last_num = last_num + 1
            speaker = message.get("role").lower()
            content = message.get("content")
            character = SystemMessage(content=content)
            if speaker == settings.VOICE_NAME_HUMAN.lower():
                character = HumanMessage(content=content)
                memory.chat_memory.add_user_message(content)
            elif speaker == settings.VOICE_NAME_AI.lower():
                character = AIMessage(content=content)
                memory.chat_memory.add_ai_message(content)
            append_to_full_conversation_history(character)
    else:
        st.error(f"{conversation} not found.")


def display_context_window_panel(memory):
    if st.session_state.memory_type.lower() == "zep":
        view_context_window_memory = st.expander("View Zep Context Window Memory")
        with view_context_window_memory:
            view_context_window_memory.json(memory.chat_memory.zep_messages)
    else:
        view_context_window_memory = st.expander("View Context Window Memory")
        with view_context_window_memory:
            view_context_window_memory.json(memory.chat_memory.messages)


def display_entire_conversation_history_panel():
    messages = get_full_conversation_history()

    view_entire_conversation_history = st.expander("View Entire Conversation")
    with view_entire_conversation_history:
        for message in messages:
            st.write(f"{get_role_from_message(message)}: {message.content}")


def display_costs_panel(costs):
    view_costs = st.sidebar.expander("Costs History")

    with view_costs:
        st.markdown(
            f"## Costs\n**Total cost: ${sum(costs):.5f}**\n"
            + "\n".join([f"- ${cost:.5f}" for cost in costs])
        )


def display_conversation_history_panel(memory, save_snapshot):

    if not os.path.exists(settings.CONVERSATION_SAVE_FOLDER):
        os.makedirs(settings.CONVERSATION_SAVE_FOLDER)

    conversation_history_panel = st.expander("Conversations History")
    with conversation_history_panel:
        conversations = [
            None,
        ] + [
            c
            for c in helper_module.get_filenames(settings.CONVERSATION_SAVE_FOLDER)
            if c.endswith("json")
        ]
        selected_conversation = st.selectbox("Conversations:", conversations)
        if st.button("Export Conversation", key="export_conversation"):
            filename = export_conversation()
            st.success(f"✏️ Conversation exported to {filename}")
        if selected_conversation:
            load_conversation(
                memory, f"{settings.CONVERSATION_SAVE_FOLDER}/{selected_conversation}"
            )
        if st.button("Load Latest Snapshot", key="load_snapshot"):
            load_conversation(
                memory, f"{settings.CONVERSATION_SAVE_FOLDER}/snapshot.json"
            )
            st.experimental_rerun()
        if (
            st.button("New Conversation", key="new_conversation")
            or len(get_messages_from_memory(memory)) == 0
        ):
            memory.chat_memory.messages.clear()
            set_custom_instructions(get_custom_instructions(default=True))
        if st.button("Rerun", key="rerun"):
            st.experimental_rerun()
        st.info(f"Last user input: {load_user_input_from_file()}")

    st.markdown(""" [Go to top](#top) """, unsafe_allow_html=True)
    st.subheader("", anchor="bottom")
    if save_snapshot:
        export_conversation(snapshot=True)


def is_agent_query(user_input):
    return any(user_input.lower().startswith(prefix) for prefix in settings.AGENT_PROMPT_PREFIXES)


# An agent method should follow the naming convention: get_<agent_name_prefix>_agent
# The agent name prefix should be the same as the agent name in the settings.py file
def get_agent_from_user_input(user_input):
    agent = None
    for prefix in settings.AGENT_PROMPT_PREFIXES:
        if user_input.lower().startswith(prefix):
            agent_method = getattr(agents, f"get_{prefix.lower().strip(':')}_agent")
            agent = agent_method(agents.get_agent_llm())
            user_input = user_input[len(prefix):]
            break
    return prefix, agent, user_input


def handle_user_input(user_input, last_num):
    system_input = get_custom_instructions()
    if user_input.lower().startswith(settings.PROMPT_KEYWORD_PREFIX_CI):
        system_input = user_input[len(settings.PROMPT_KEYWORD_PREFIX_CI):].strip()
        set_custom_instructions(system_input)
    elif user_input.lower().startswith(settings.PROMPT_KEYWORD_PREFIX_SYSTEM):
        system_input = user_input[len(settings.PROMPT_KEYWORD_PREFIX_SYSTEM):].strip()
    elif is_agent_query(user_input):
        handle_message(HumanMessage(content=user_input.strip()), last_num + 1)
        append_to_full_conversation_history(HumanMessage(content=user_input.strip()))
    else:
        handle_message(HumanMessage(content=user_input.strip()), last_num + 1)
        append_to_full_conversation_history(HumanMessage(content=user_input.strip()))
    if user_input.lower().startswith(settings.PROMPT_KEYWORD_PREFIX_CI) or user_input.lower().startswith(settings.PROMPT_KEYWORD_PREFIX_SYSTEM):
        handle_message(SystemMessage(content=system_input), last_num + 1)
    return system_input


def display_cost_info(cb, answer, memory):
    if st.session_state.streaming_enabled:
        model_cost_mapping = {
            "gpt-4": (
                settings.MODEL_COST_GPT4_8K_INPUT,
                settings.MODEL_COST_GPT4_8K_OUTPUT,
            ),
            "gpt-3.5-turbo-16k": (
                settings.MODEL_COST_GPT3_TURBO_16K_INPUT,
                settings.MODEL_COST_GPT3_TURBO_16K_OUTPUT,
            ),
            "gpt-3.5-turbo": (
                settings.MODEL_COST_GPT3_TURBO_4K_INPUT,
                settings.MODEL_COST_GPT3_TURBO_4K_OUTPUT,
            ),
        }

        input_token_cost, output_token_cost = model_cost_mapping.get(
            st.session_state.model_name, (0, 0)
        )

        prompt_tokens = token_counter.token_counter(
            convert_messages_to_str(get_messages_from_memory(memory))
        )
        comp_tokens = token_counter.token_counter(answer)
        total_cost = (prompt_tokens * input_token_cost) + (
            comp_tokens * output_token_cost
        )

        st.caption(f"Streaming enabled. Costs are approximation.")
        st.caption(f"Total Tokens: {prompt_tokens + comp_tokens}")
        st.caption(f"Prompt Tokens: {prompt_tokens}")
        st.caption(f"Completion Tokens: {comp_tokens}")
        st.caption(f"Total Cost: ${total_cost}")

        cost = total_cost

    else:
        cost = cb.total_cost

        helper_module.log(cb, "info")
        st.caption(f"Total Tokens: {cb.total_tokens}")
        st.caption(f"Prompt Tokens: {cb.prompt_tokens}")
        st.caption(f"Completion Tokens: {cb.completion_tokens}")
        st.caption(f"Total Cost: ${cb.total_cost}")

    return cost


def load_user_input_from_file():
    user_input = helper_module.read_text_file(settings.USER_INPUT_SAVE_FILE)
    if user_input:
        return user_input
    else:
        return ""


def setup_and_cleanup(func):
    # TODO: pre-processing

    helper_module.log(
        f"------------ {characters.AI_NAME} v{settings.VERSION} running... ------------", "info"
    )
    helper_module.log(
        f"------------ Initializing... ------------", "info"
    )

    # make folders if they don't exist
    for folder in settings.FOLDERS_TO_MAKE:
        Path(folder).mkdir(parents=True, exist_ok=True)

    # Flask runs on port 5000
    is_server_running = os.system(f"lsof -i :{settings.AUDIO_SERVER_PORT}")

    if is_server_running != 0:
        helper_module.log(
            f"Firing up Flask audio server...", "info"
        )
        subprocess.Popen(["python", "audio_server.py"])
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY')

    helper_module.log(
        f"Initializing Streamlit session state...", "info"
    )

    init_session_state()

    def wrapper(*args, **kwargs):
        func(*args, **kwargs)

    # TODO: post-processing

    helper_module.log(
        f"------------ Cleaning up... ------------", "info"
    )

    return wrapper


# pre- and post- processor decorator for main function
@setup_and_cleanup
def main():
    init_page()
    display_tts_panel()
    save_snapshot = False

    # Chat message history that stores messages in Streamlit session state. Remember it works as context window and doesn't retain the whole conversation history!

    old_context_window = StreamlitChatMessageHistory("context_window")
    new_context_window = update_context_window(old_context_window)
    ai_model = select_model(memory=new_context_window)

    helper_module.log(f"Memory type: {st.session_state.memory_type}", "info")

    display_context_window_panel(new_context_window)
    display_entire_conversation_history_panel()

    last_num = len(get_full_conversation_history())
    for i, message in enumerate(get_full_conversation_history(), start=1):
        handle_message(message, i)
    user_input = (
        st.session_state.pop("transcript", "")
        if st.session_state.use_audio
        else st.chat_input("Prompt: ")
    )

    available_prefix_options = ["None"] + settings.ALL_PROMPT_PREFIXES

    prefix_option = st.selectbox(
        "Keyword Prefix",
        available_prefix_options)

    if user_input:
        if prefix_option != "None":
            user_input = f"{prefix_option} {user_input}"
            helper_module.log(f"Keyword prefix used: {prefix_option} {user_input}", "info")
        # try saving last user input in case of runtime error
        save_user_input_to_file(user_input)
        system_input = handle_user_input(user_input, last_num)
        if system_input:
            with (st.spinner("Pippa is typing ...")):
                with get_openai_callback() as cb:
                    stream_handler = StreamHandler(st.empty())
                    if user_input.lower().startswith(settings.PROMPT_KEYWORD_PREFIX_QA):
                        memory = ConversationBufferMemory(input_key="question",
                                                          memory_key="history")
                        helper_module.log(f"Retrieval QA Session Started...: {user_input}", "info")
                        display_vectordb_info()
                        answer, docs = retrieval_qa_run(system_input, user_input, memory, callbacks=[stream_handler])
                        if settings.SHOW_SOURCES:
                            helper_module.log("----------------------------------SOURCE DOCUMENTS BEGIN---------------------------")
                            for document in docs:
                                helper_module.log("\n> " + document.metadata["source"] + ":")
                                helper_module.log(document.page_content)
                            helper_module.log("----------------------------------SOURCE DOCUMENTS END---------------------------")
                            new_context_window.save_context({"human_input": user_input}, {"output": answer})
                    elif is_agent_query(user_input):
                        helper_module.log(f"Agent Session Started...: {user_input}", "info")
                        prefix, agent, user_input = get_agent_from_user_input(user_input)
                        intermediate_answer = agent(user_input)["output"]
                        helper_module.log(f"Agent: {prefix}", "info")
                        helper_module.log(f"Normal Chat Session Started...: {user_input}", "info")
                        helper_module.log(f"User message: {user_input}", "info")
                        helper_module.log(f"Intermediate answer: {intermediate_answer}", "info")

                        if prefix == settings.PROMPT_KEYWORD_PREFIX_DALLE:
                            pattern = r'\((.*?)\)'
                            matches = re.findall(pattern, intermediate_answer)
                            image_url = matches[0] if matches else None
                            markdown_string = f'<a href="{image_url}" target="_blank"><img src="{image_url}" width="{settings.DALLE_IMAGE_SCALE_FACTOR}"/></a>'
                            st.markdown(markdown_string, unsafe_allow_html=True)
                        st.subheader("""Intermediate Answer from Agent""")
                        st.markdown(intermediate_answer)

                        system_input = system_input + " No matter what the user asked, you must give this answer exactly as it is including the markdown formatting in the language the user asked: " + intermediate_answer
                        helper_module.log(f"System message: {system_input}", "info")
                        answer = ai_model.run(
                            system_input=system_input,
                            human_input=user_input,
                            callbacks=[stream_handler],
                        )
                        new_context_window.save_context({"human_input": user_input}, {"output": answer})
                    else:
                        helper_module.log(f"Normal Chat Session Started...: {user_input}", "info")
                        answer = ai_model.run(
                            system_input=system_input,
                            human_input=user_input,
                            callbacks=[stream_handler],
                        )

                    handle_message(AIMessage(content=answer), last_num + 2)
                    append_to_full_conversation_history(AIMessage(content=answer))

        save_snapshot = True
        new_cost = display_cost_info(cb, answer, new_context_window)
        st.session_state.costs.append(new_cost)

    display_costs_panel(st.session_state.get("costs", []))
    display_conversation_history_panel(new_context_window, save_snapshot)


if __name__ == "__main__":
    main()
