import re
from time import sleep

from langchain import ConversationChain
from langchain.memory import ConversationBufferMemory
from streamlit_chat import message

import utils

auto_user_memory = ConversationBufferMemory()


def get_conversation_chain(llm):
    return ConversationChain(
        llm=llm,
        verbose=False,
        memory=auto_user_memory,
        prompt=utils.prompt_user
    )


def auto_user_loop(st, user_input, generate_response, generate_auto_user):
    while True:
        st.session_state.past.append(user_input)
        user_input = generate_response(user_input)
        st.session_state.generated.append(user_input)
        user_input = generate_auto_user(user_input)
        st.session_state.past.append(user_input)
        if st.session_state['generated']:
            message(st.session_state['past'][st.session_state.index], is_user=True, key=str(st.session_state.index) + '_auto_user')
            message(st.session_state["generated"][st.session_state.index], key=str(st.session_state.index) + '_auto')
            st.session_state.index += 1
        sleep(1)
        if re.search(r"\bend\b", user_input, re.IGNORECASE):
            print('finish', user_input)
            break
    return 'Thank you for using our chatbot!'
