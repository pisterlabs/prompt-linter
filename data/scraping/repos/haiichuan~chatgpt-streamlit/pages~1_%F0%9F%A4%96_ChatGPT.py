import time

import openai
import pandas as pd
import streamlit as st

from utils import load_prompt_templates, load_prompts, render_footer, render_github_info

st.set_page_config(page_title="ChatGPT Web", page_icon="🤖")


@st.cache_resource
def init_openai_settings():
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    if st.secrets.get("OPENAI_PROXY"):
        openai.proxy = st.secrets["OPENAI_PROXY"]


def init_session():
    if not st.session_state.get("params"):
        st.session_state["params"] = dict()
    if not st.session_state.get("chats"):
        st.session_state["chats"] = {}
    if "input" not in st.session_state:
        st.session_state["input"] = "Hello, how are you!"


def new_chat(chat_name):
    if not st.session_state["chats"].get(chat_name):
        st.session_state["chats"][chat_name] = {
            "answer": [],
            "question": [],
            "messages": [
                {"role": "system", "content": st.session_state["params"]["prompt"]}
            ],
            "is_delete": False,
            "display_name": chat_name,
        }
    return chat_name


def switch_chat(chat_name):
    if st.session_state.get("current_chat") != chat_name:
        st.session_state["current_chat"] = chat_name
        render_chat(chat_name)
        st.stop()


def switch_chat_name(chat_name):
    if st.session_state.get("current_chat") != chat_name:
        st.session_state["current_chat"] = chat_name
        render_sidebar()
        render_chat(chat_name)
        st.stop()


def delete_chat(chat_name):
    if chat_name in st.session_state['chats']:
        st.session_state['chats'][chat_name]['is_delete'] = True

    current_chats = [chat for chat, value in st.session_state['chats'].items() if not value['is_delete']]
    if len(current_chats) == 0:
        switch_chat(new_chat(f"Chat{len(st.session_state['chats'])}"))
        st.stop()

    if st.session_state["current_chat"] == chat_name:
        del st.session_state["current_chat"]
        switch_chat_name(current_chats[0])


def edit_chat(chat_name, zone):
    def edit():
        if not st.session_state['edited_name']:
            print('name is empty!')
            return None

        if (st.session_state['edited_name'] != chat_name
                and st.session_state['edited_name'] in st.session_state['chats']):
            print('name is duplicated!')
            return None

        if st.session_state['edited_name'] == chat_name:
            print('name is not modified!')
            return None

        st.session_state['chats'][chat_name]['display_name'] = st.session_state['edited_name']

    edit_zone = zone.empty()
    time.sleep(0.1)
    with edit_zone.container():
        st.text_input('New Name', st.session_state['chats'][chat_name]['display_name'], key='edited_name')
        column1, _, column2 = st.columns([1, 5, 1])
        column1.button('✅', on_click=edit)
        column2.button('❌')


def render_sidebar_chat_management(zone):
    new_chat_button = zone.button(label="➕ New Chat", use_container_width=True)
    if new_chat_button:
        new_chat_name = f"Chat{len(st.session_state['chats'])}"
        st.session_state["current_chat"] = new_chat_name
        new_chat(new_chat_name)

    with st.sidebar.container():
        for chat_name in st.session_state["chats"].keys():
            if st.session_state['chats'][chat_name]['is_delete']:
                continue
            if chat_name == st.session_state.get('current_chat'):
                column1, column2, column3 = zone.columns([7, 1, 1])
                column1.button(
                    label='💬 ' + st.session_state['chats'][chat_name]['display_name'],
                    on_click=switch_chat_name,
                    key=chat_name,
                    args=(chat_name,),
                    type='primary',
                    use_container_width=True,
                )
                column2.button(label='📝', key='edit', on_click=edit_chat, args=(chat_name, zone))
                column3.button(label='🗑️', key='remove', on_click=delete_chat, args=(chat_name,))
            else:
                zone.button(
                    label='💬 ' + st.session_state['chats'][chat_name]['display_name'],
                    on_click=switch_chat_name,
                    key=chat_name,
                    args=(chat_name,),
                    use_container_width=True,
                )

    if new_chat_button:
        switch_chat(new_chat_name)


def render_sidebar_gpt_config_tab(zone):
    st.session_state["params"] = dict()
    st.session_state["params"]["model"] = zone.selectbox(
        "Please select a model",
        ["gpt-3.5-turbo"],  # , "gpt-4"
        help="ID of the model to use",
    )
    st.session_state["params"]["temperature"] = zone.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=1.2,
        step=0.1,
        format="%0.2f",
        help="What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.",
    )
    st.session_state["params"]["max_tokens"] = zone.slider(
        "Max Tokens",
        value=2000,
        step=1,
        min_value=100,
        max_value=2000,
        help="The maximum number of tokens to generate in the completion",
    )
    st.session_state["params"]["stream"] = zone.checkbox(
        "Steaming output",
        value=True,
        help="If set, partial message deltas will be sent, like in ChatGPT. Tokens will be sent as data-only server-sent events as they become available, with the stream terminated by a data: [DONE] message",
    )
    zone.caption('Looking for help at https://platform.openai.com/docs/api-reference/chat')


def render_sidebar_prompt_config_tab(zone):
    prompt_text = zone.empty()
    st.session_state["params"]["prompt"] = prompt_text.text_area(
        "System Prompt",
        "You are a helpful assistant that answer questions as possible as you can.",
        help="The prompt(s) to generate completions for, encoded as a string, array of strings, array of tokens, or array of token arrays.",
    )
    template = zone.selectbox('Loading From Prompt Template', load_prompt_templates())
    if template:
        prompts_df = load_prompts(template)
        actor = zone.selectbox('Loading Prompts', prompts_df.index.tolist())
        if actor:
            st.session_state["params"]["prompt"] = prompt_text.text_area(
                "System Prompt",
                prompts_df.loc[actor].prompt,
                help="The prompt(s) to generate completions for, encoded as a string, array of strings, array of tokens, or array of token arrays.",
            )


def render_download_zone(zone):
    from io import BytesIO, StringIO
    if not st.session_state.get('current_chat'):
        return

    chat = st.session_state['chats'][st.session_state['current_chat']]
    col1, col2 = zone.columns([1, 1])

    chat_messages = ['# ' + chat['display_name']]
    if chat["question"]:
        for i in range(len(chat["question"])):
            chat_messages.append(f"""😃 **YOU:** {chat["question"][i]}""")
            if i < len(chat["answer"]):
                chat_messages.append(f"""🤖 **AI:** {chat["answer"][i]}""")
        col1.download_button('📤 Markdown', '\n'.join(chat_messages).encode('utf-8'),
                             file_name=f"{chat['display_name']}.md", help="Download messages to a markdown file",
                             use_container_width=True)

    tables = []
    for answer in chat["answer"]:
        filter_table_str = '\n'.join([m.strip() for m in answer.split('\n') if m.strip().startswith('| ') or m == ''])
        try:
            tables.extend(
                [pd.read_table(StringIO(filter_table_str.replace(' ', '')), sep='|').dropna(axis=1, how='all').iloc[1:]
                 for m in filter_table_str.split('\n\n')])
        except Exception as e:
            print(e)
    if tables:
        buffer = BytesIO()
        with pd.ExcelWriter(buffer) as writer:
            for index, table in enumerate(tables):
                table.to_excel(writer, sheet_name=str(index + 1), index=False)
        col2.download_button('📉 Excel', buffer.getvalue(), file_name=f"{chat['display_name']}.xlsx",
                             help="Download tables to a excel file", use_container_width=True)


def render_sidebar():
    chat_name_container = st.sidebar.container()
    chat_config_expander = st.sidebar.expander('⚙️ Chat configuration', True)
    tab_gpt, tab_prompt = chat_config_expander.tabs(['🌐     ChatGPT', '👥 Prompt'])
    download_zone = st.sidebar.empty()
    github_zone = st.sidebar.empty()

    render_sidebar_gpt_config_tab(tab_gpt)
    render_sidebar_prompt_config_tab(tab_prompt)
    render_sidebar_chat_management(chat_name_container)
    render_download_zone(download_zone)
    render_github_info(github_zone)


def render_user_message(message, zone):
    col1, col2 = zone.columns([1, 8])
    col1.markdown("😃 **YOU:**")
    col2.markdown(message)


def render_ai_message(message, zone):
    col1, col2 = zone.columns([1, 8])
    col1.markdown("🤖 **AI:**")
    col2.markdown(message)


def render_history_answer(chat, zone):
    zone.empty()
    time.sleep(0.1)  # https://github.com/streamlit/streamlit/issues/5044
    with zone.container():
        if chat['messages']:
            st.caption(f"""ℹ️ Prompt: {chat["messages"][0]['content']}""")
        if chat["question"]:
            for i in range(len(chat["question"])):
                render_user_message(chat["question"][i], st)
                if i < len(chat["answer"]):
                    render_ai_message(chat["answer"][i], st)


def render_last_answer(question, chat, zone):
    answer_zone = zone.empty()

    chat["messages"].append({"role": "user", "content": question})
    chat["question"].append(question)
    if st.session_state["params"]["stream"]:
        answer = ""
        chat["answer"].append(answer)
        chat["messages"].append({"role": "assistant", "content": answer})
        for response in get_openai_response(chat["messages"]):
            answer += response["choices"][0]['delta'].get("content", '')
            chat["answer"][-1] = answer
            chat["messages"][-1] = {"role": "assistant", "content": answer}
            render_ai_message(answer, answer_zone)
    else:
        with st.spinner("Wait for responding..."):
            answer = get_openai_response(chat["messages"])["choices"][0]["message"]["content"]
            chat["answer"].append(answer)
            chat["messages"].append({"role": "assistant", "content": answer})
            render_ai_message(answer, zone)


def render_stop_generate_button(zone):
    def stop():
        st.session_state['regenerate'] = False

    zone.columns((2, 1, 2))[1].button('□ Stop', on_click=stop)


def render_regenerate_button(chat, zone):
    def regenerate():
        chat["messages"].pop(-1)
        chat["messages"].pop(-1)
        chat["answer"].pop(-1)
        st.session_state['regenerate'] = True
        st.session_state['last_question'] = chat["question"].pop(-1)

    zone.columns((2, 1, 2))[1].button('🔄Regenerate', type='primary', on_click=regenerate)


def render_chat(chat_name):
    def handle_ask():
        if st.session_state['input']:
            re_generate_zone.empty()
            render_user_message(st.session_state['input'], last_question_zone)
            render_stop_generate_button(stop_generate_zone)
            render_last_answer(st.session_state['input'], chat, last_answer_zone)
            st.session_state['input'] = ''

    if chat_name not in st.session_state["chats"]:
        st.error(f'{chat_name} is not exist')
        return
    chat = st.session_state["chats"][chat_name]
    if chat['is_delete']:
        st.error(f"{chat_name} is deleted")
        st.stop()
    if len(chat['messages']) == 1 and st.session_state["params"]["prompt"]:
        chat["messages"][0]['content'] = st.session_state["params"]["prompt"]

    conversation_zone = st.container()
    history_zone = conversation_zone.empty()
    last_question_zone = conversation_zone.empty()
    last_answer_zone = conversation_zone.empty()
    ask_form_zone = st.empty()

    render_history_answer(chat, history_zone)

    ask_form = ask_form_zone.form(chat_name)
    col1, col2 = ask_form.columns([10, 1])
    col1.text_area("😃 You: ",
                   key="input",
                   max_chars=2000,
                   label_visibility='collapsed')

    with col2.container():
        for _ in range(2):
            st.write('\n')
        st.form_submit_button("🚀", on_click=handle_ask)
    stop_generate_zone = conversation_zone.empty()
    re_generate_zone = conversation_zone.empty()

    if st.session_state.get('regenerate'):
        render_user_message(st.session_state['last_question'], last_question_zone)
        render_stop_generate_button(stop_generate_zone)
        render_last_answer(st.session_state['last_question'], chat, last_answer_zone)
        st.session_state['regenerate'] = False

    if chat["answer"]:
        stop_generate_zone.empty()
        render_regenerate_button(chat, re_generate_zone)

    render_footer()


def get_openai_response(messages):
    if st.session_state["params"]["model"] in {'gpt-3.5-turbo', 'gpt4'}:
        response = openai.ChatCompletion.create(
            model=st.session_state["params"]["model"],
            temperature=st.session_state["params"]["temperature"],
            messages=messages,
            stream=st.session_state["params"]["stream"],
            max_tokens=st.session_state["params"]["max_tokens"],
        )
    else:
        raise NotImplementedError('Not implemented yet!')
    return response


if __name__ == "__main__":
    init_openai_settings()
    init_session()
    render_sidebar()
    if st.session_state.get("current_chat"):
        render_chat(st.session_state["current_chat"])
    if len(st.session_state["chats"]) == 0:
        switch_chat(new_chat(f"Chat{len(st.session_state['chats'])}"))
