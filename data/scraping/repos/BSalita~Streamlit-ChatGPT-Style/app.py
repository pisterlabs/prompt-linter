import openai
import streamlit as st
import streamlit_chat

openai.api_key = st.secrets["OPENAI_API_KEY"]

# be sure to end each prompt string with a comma.
example_user_prompts = [
    "echo Hello World!",
    "How old is Elon Musk?",
    "What makes a good joke?",
    "Tell me a haiku.",
]


def move_focus():
    # inspect the html to determine which control to specify to receive focus (e.g. text or textarea).
    st.components.v1.html(
        f"""
            <script>
                var textarea = window.parent.document.querySelectorAll("textarea[type=textarea]");
                for (var i = 0; i < textarea.length; ++i) {{
                    textarea[i].focus();
                }}
            </script>
        """,
    )


def stick_it_good():

    # make header sticky.
    st.markdown(
        """
            <div class='fixed-header'/>
            <style>
                div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
                    position: sticky;
                    top: 2.875rem;
                    background-color: white;
                    z-index: 999;
                }
                .fixed-header {
                    border-bottom: 1px solid black;
                }
            </style>
        """,
        unsafe_allow_html=True
    )


def userid_change():
    st.session_state.userid = st.session_state.userid_input
    
    
def complete_messages(nbegin,nend,stream=False):
    messages = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]
    with st.spinner(f"Waiting for {nbegin}/{nend} responses from ChatGPT."):
        if stream:
            responses = []
            # Looping over openai's responses. async style.
            for response in openai.ChatCompletion.create(
                model = st.session_state["openai_model"],
                messages = messages,
                stream = True):
                    partial_response_content = response.choices[0].delta.get("content","")
                    responses.append(partial_response_content)
            response_content = "".join(responses)
        else:
            response = openai.ChatCompletion.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=False,
            )
            response_content = response.choices[0]['message'].get("content","")
    return response_content
    

def main():
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    with st.container():
        st.title("Streamlit ChatGPT Bot")
        stick_it_good()

    if "userid" in st.session_state:
        st.sidebar.text_input(
            "Current userid", on_change=userid_change, placeholder=st.session_state.userid, key='userid_input')
        if st.sidebar.button("Clear Conversation", key='clear_chat_button'):
            st.session_state.messages = []
            move_focus()
        if st.sidebar.button("Show Example Conversation", key='show_example_conversation'):
            #st.session_state.messages = [] # don't clear current conversaations?
            for i,up in enumerate(example_user_prompts):
                st.session_state.messages.append({"role": "user", "content": up})
                assistant_content = complete_messages(i,len(example_user_prompts))
                st.session_state.messages.append({"role": "assistant", "content": assistant_content})
            move_focus()
        for i,message in enumerate(st.session_state.messages):
            nkey = int(i/2)
            if message["role"] == "user":
                streamlit_chat.message(message["content"], is_user=True, key='chat_messages_user_'+str(nkey))
            else:
                streamlit_chat.message(message["content"], is_user=False, key='chat_messages_assistant_'+str(nkey))

        if user_content := st.chat_input("Type your question here."): # using streamlit's st.chat_input because it stays put at bottom, chat.openai.com style.
                nkey = int(len(st.session_state.messages)/2)
                streamlit_chat.message(user_content, is_user=True, key='chat_messages_user_'+str(nkey))
                st.session_state.messages.append({"role": "user", "content": user_content})
                assistant_content = complete_messages(0,1)
                streamlit_chat.message(assistant_content, key='chat_messages_assistant_'+str(nkey))
                st.session_state.messages.append({"role": "assistant", "content": assistant_content})
                #len(st.session_state.messages)
    else:
        st.sidebar.text_input(
            "Enter a random userid", on_change=userid_change, placeholder='userid', key='userid_input')
        streamlit_chat.message("Hi. I'm your friendly streamlit ChatGPT assistant.",key='intro_message_1')
        streamlit_chat.message("To get started, enter a random userid in the left sidebar.",key='intro_message_2')
                
if __name__ == '__main__':
    main()

