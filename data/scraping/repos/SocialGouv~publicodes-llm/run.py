import sys
import logging
import streamlit as st
import random

from LlamaIndexFormatter import LlamaIndexFormatter
from tool import PublicodeAgent

#
# this is "just" the streamlit UI wrapper around the llama_index agent
#

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

logger = logging.getLogger()

handler = logging.StreamHandler(stream=sys.stdout)
handler.setFormatter(LlamaIndexFormatter())
logger.addHandler(handler)

st.set_page_config(
    page_title="LLM + publicodes = ❤️",
    page_icon="🐫",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)
st.header("LLM + publicodes = ❤️")
# st.title(
#     "Interrogez le modèle publicodes de calcul de préavis de retraite (expérimental)"
# )
st.info(
    """
Interrogez le modèle publicodes de calcul de préavis de retraite (expérimental)

Si on vous demande la convention collective, répondre IDCC1979 ou IDCC1043 par exemple (WIP)

Exemple : Quel est mon préavis de retraite ?

Les détails [sur GitHub](https://github.com/SocialGouv/publicodes-llm)""".format(),
    icon="💡",
)


if "messages" not in st.session_state.keys():  # Initialize the chat message history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Bienvenue !",
        }
    ]


# chat_engine = index.as_chat_engine(
#     chat_mode="context", verbose=True, similarity_top_k=5
# )


if prompt := st.chat_input("A votre écoute :)"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# prompt = "Peux tu me calculer mon préavis de retraite ?"

for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# if not st.session_state.messages:
#     st.session_state.messages.append(
#         {
#             "role": "user",
#             "content": "Peux tu me calculer mon préavis de retraite ?",
#         }
#     )


# if not st.session_state.messages:

#     st.session_state.messages.append(
#         {"role": "assistant", "content": "Comment puis-je vous aider ?"}
#     )

if "agent" not in st.session_state:
    # set the initial default value of the slider widget
    agent = PublicodeAgent()
    st.session_state["agent"] = agent

waiters = [
    "Je refléchis...",
    "Hummmm laissez moi chercher...",
    "Je cherche des réponses...",
]

if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.spinner(random.choice(waiters)):
            # print("prompt", prompt)
            message_placeholder = st.empty()
            if prompt and st.session_state.agent:
                print("hello", prompt)
                streaming_response = st.session_state.agent.chat(prompt)

                # streaming_response.print_response_stream()
                full_response = streaming_response
                message_placeholder.markdown(streaming_response)
                # for text in streaming_response.response_gen:
                #     full_response += text
                #     message_placeholder.markdown(full_response)

                if full_response:
                    st.session_state.messages.append(
                        {"role": "assistant", "content": full_response.response}
                    )
