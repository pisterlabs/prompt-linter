import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain.callbacks import get_openai_callback

def init_page():
    st.set_page_config(
        page_title="Mein persönlicher ChatBot",
    )
    st.header("Mein persönlicher ChatBot")
    st.sidebar.title("Optionen")


def init_messages():
    clear_button = st.sidebar.button("Konversation löschen", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="Du bist ein hilreicher Assisstent. Du antwortest immer in Deutsche, es denn Du wirst dazu aufgefordert.")
        ]
        st.session_state.costs = []


def select_model():
    model = st.sidebar.radio("Wähle ein LLM:", ("GPT-3.5", "GPT-4"))
    if model == "GPT-3.5":
        model_name = "gpt-3.5-turbo"
    else:
        model_name = "gpt-4"

    # Add a slider to allow users to select the temperature from 0 to 2.
    # The initial value should be 0.0, with an increment of 0.01.
    temperature = st.sidebar.slider("Temperatur:", min_value=0.0, max_value=2.0, value=0.0, step=0.01)

    return ChatOpenAI(temperature=temperature, model_name=model_name)


def get_answer(llm, messages):
    with get_openai_callback() as cb:
        answer = llm(messages)
    return answer.content, cb.total_cost


def main():
    init_page()

    llm = select_model()
    init_messages()

    # Monitor user input
    if user_input := st.chat_input("Gib Deine Frage hier ein:"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("ChatGPT arbeitet und schreibt ..."):
            answer, cost = get_answer(llm, st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=answer))
        st.session_state.costs.append(cost)

    messages = st.session_state.get('messages', [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message('assistant'):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message('user'):
                st.markdown(message.content)
        else:  # isinstance(message, SystemMessage):
            st.write(f"System message: {message.content}")

    costs = st.session_state.get('costs', [])
    st.sidebar.markdown("## Kosten")
    st.sidebar.markdown(f"**Summe Kosten: ${sum(costs):.5f}**")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")

if __name__ == '__main__':
    main()
