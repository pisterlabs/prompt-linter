# Only need to change this for each character/page
import streamlit as st

# from typing import Literal, LiteralString
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from prompts.custom_few_shot import CustomFewShotChatMessagePromptTemplate
from persona_example_selector import PersonaExampleSelector
from prompt_templates import (
    MAIN_TEMPLATE,
    SYSTEM_NOTE_TEMPLATE,
    NSFW_TEMPLATE,
    TASK_TEMPLATE,
    EXAMPLES_PREFACE_TEMPLATE,
)

# the rest
from persona_ids import PERSONA_IDS


def page_template(character_data):
    st.set_page_config(
        page_title="Mimicking a Character via Explicit and Implicit Instructions",
    )

    character_name = character_data["identified_character"]
    character_first_name = character_name.split(" ")[0]
    persona_id = PERSONA_IDS[character_name]

    debug_info_key = f"debug_info_{persona_id}"
    messages_key = f"langchain_messages_{persona_id}"

    if debug_info_key not in st.session_state:
        st.session_state[debug_info_key] = ""

    st.title(character_name)
    st.caption("Mimicking a Character via Explicit and Implicit Instructions")

    # this is a hack, but streamlit isn't properly sharing the state data that that is bound to the input widgets on the home page, so I am creating a copy and assigning it while it is set
    if "openai_api_key_p" not in st.session_state:
        st.session_state["openai_api_key_p"] = st.session_state.openai_api_key

    if "user_name_p" not in st.session_state:
        st.session_state["user_name_p"] = st.session_state.user_name

    with st.expander("personality assessment", expanded=False):
        st.json(character_data)

    view_info = st.expander("retrieval debug information", expanded=False)

    msgs = StreamlitChatMessageHistory(key=messages_key)
    memory = ConversationBufferMemory(
        chat_memory=msgs,
        ai_prefix=character_first_name,
        human_prefix=st.session_state.user_name_p,
        input_key="human_input",
        return_messages=True,
    )

    example_selector = PersonaExampleSelector(
        persona_id=persona_id,
    )

    examples_prompt = CustomFewShotChatMessagePromptTemplate(
        example_selector=example_selector,
        example_prompt=ChatPromptTemplate.from_messages(
            [("human", "{cue}"), ("ai", "{response}")]
        ),
        single_prompt=ChatPromptTemplate.from_messages([("ai", "{text}")]),
        system_prompt=ChatPromptTemplate.from_messages([("system", "{text}")]),
        input_variables=["human_input"],
    )

    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_NOTE_TEMPLATE),
            ("system", NSFW_TEMPLATE),
            ("system", MAIN_TEMPLATE.format(**character_data)),
            (
                "system",
                EXAMPLES_PREFACE_TEMPLATE.format(
                    user_name=st.session_state.user_name_p,
                    character_name=character_first_name,
                ),
            ),
            examples_prompt,
            (
                "system",
                TASK_TEMPLATE.format(
                    character_name=character_first_name,
                    user_name=st.session_state.user_name_p,
                ),
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{human_input}"),
        ]
    )

    llm_chain = LLMChain(
        llm=ChatOpenAI(
            openai_api_key=st.session_state.openai_api_key_p,
            model="gpt-4",
            temperature=0.9,
            model_kwargs={
                "frequency_penalty": 0.7,
                "presence_penalty": 0.7,
            },
        ),
        prompt=chat_prompt,
        memory=memory,
        verbose=True,
    )

    character_avatar = f"img/{character_name.lower().replace(' ', '_')}.png"
    print(character_avatar)
    user_avatar = "img/anon.png"

    for msg in msgs.messages:
        # AVATAR: LiteralString | Literal['img/anon.png'] = character_avatar if msg.type == "ai" else user_avatar
        AVATAR = character_avatar if msg.type == "ai" else user_avatar
        st.chat_message(msg.type, avatar=AVATAR).write(msg.content)

    if prompt := st.chat_input():
        st.chat_message("human", avatar=user_avatar).write(prompt)
        response = llm_chain.run(prompt)
        st.chat_message("ai", avatar=character_avatar).write(response)

    with view_info:
        st.code(st.session_state[debug_info_key])
