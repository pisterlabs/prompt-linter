import openai
import streamlit as st
import os
import quiz_utils

DEV_MODE = os.environ.get("DEV") == "1"

def write_message(st, role: str, content: str, remember: bool = True, show: bool = True):
    assert role in ["user", "assistant"], f"Invalid role {role}"
    msg = {"role": role, "content": content}
    if show:
        st.chat_message(role).write(content)
    if remember:
        st.session_state.messages.append(msg)
    assert True in [remember, show], "write_message called with remember=False and show=False, this is probably a mistake"

# Initialize session state variable if not present
if "mode" not in st.session_state:
    st.session_state["mode"] = None # | "quiz" | "lesson"

if "button_clicked" not in st.session_state:
    st.session_state["button_clicked"] = False
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

openai_api_key = os.environ.get("OPENAI_KEY")
if openai_api_key is None:
    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
        "[View the source code](https://github.com/TheGrandNobody/CLT-Hackathon-2023-Team-CS)"

st.title("📝 Edion Content Generator V1")
st.caption("made by Edion")

if DEV_MODE:
    st.write("state = ", st.session_state)

# write history of messages
if st.session_state["mode"] is None:
    for msg in st.session_state.messages[:1]:
        st.chat_message(msg["role"]).write(msg["content"])

# Display buttons only if none has been clicked yet
if st.session_state["mode"] == "lesson" or st.button("Create a quiz", disabled=st.session_state["button_clicked"]):
    if st.session_state["mode"] is None:
        st.session_state["mode"] = "quiz"
        #st.session_state.messages.clear()
        msg = {"role": "assistant", "content": "Briefly explain what topic you want a quiz about, and how many questions are desired."}
        st.session_state.messages.append(msg)
        st.chat_message("assistant").write(msg["content"])
        st.session_state["button_clicked"] = True
        st.experimental_rerun()

if st.session_state["mode"] == "quiz" or st.button("Create a lesson plan", disabled=st.session_state["button_clicked"]):
    if st.session_state["mode"] is None:
        st.session_state["mode"] = "lesson"
        msg = {"role": "assistant", "content": "What topic would you like the lesson plan about? Provide all desired details."}
        st.session_state.messages.append(msg)
        st.chat_message("assistant").write(msg["content"])
        st.session_state["button_clicked"] = True
        st.experimental_rerun()

for msg in st.session_state.messages[1:]:
    st.chat_message(msg["role"]).write(msg["content"])

if st.session_state["mode"] is not None:
    if prompt := st.chat_input():
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        openai.api_key = openai_api_key
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        GENERATING_MSG = "Generating, this may take a while..."
        if st.session_state['mode'] == 'quiz':
            ### quiz generation:
            print("generating quiz...")
            st.chat_message("assistant").write(GENERATING_MSG)
            quiz_dict, new_history = quiz_utils.get_quiz_for_prompt(st.session_state.messages, openai_api_key)
            #st.session_state.messages = new_history
            if quiz_dict is None:
                st.chat_message(f"Failed to generate a valid quiz, please refresh the page and try again :(").write(quiz_dict)
                st.stop()
            else:
                write_message(st, "assistant", f"{len(quiz_dict)} question quiz is ready for download!")
                #st.chat_message("assistant").write(f"{len(quiz_dict)} question quiz is ready for download!")
                st.session_state.quiz = quiz_dict
        else:
            ### lesson plan generation:
            st.session_state.messages.append({"role": "user", "content": f"Give me the lesson plan in detail and well-formatted please"})
            print("generating lesson plan...")
            st.chat_message("assistant").write(GENERATING_MSG)

            st.session_state.messages.append({"role": "user", "content": "Also, please don't preface your messages with introductory phrases, instead be straight to the point (get to answering straight away)"})
            response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
            msg = response.choices[0].message
            st.session_state.messages.append(msg)
            st.chat_message("assistant").write(msg.content)

if "quiz" in st.session_state:
    quiz_dict = st.session_state.quiz
    st.session_state["mode"] = None # | "quiz" | "lesson"
    print(f"{len(quiz_dict)} question quiz available")
    pdf_data = quiz_utils.generate_pdf(quiz_dict)
    if st.download_button(
        label="Download PDF",
        data=pdf_data,
        file_name="quiz.pdf",
        mime="application/pdf"
    ):
        print('user downloading pdf!')