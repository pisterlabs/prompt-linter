import os
import uuid
import streamlit as st
from loguru import logger
from streamlit_js_eval import streamlit_js_eval
from streamlit_feedback import streamlit_feedback

from ai_analysis.openai_tools import openai_response, openai_response_for_text
from ai_analysis.prompts.default import assesment_default
from ai_analysis.prompts.screening import screening_assessment, scr_assessment_format
from ai_analysis.prompts.tech import tech_assessment, tech_assessment_format
from utils.st_sessionstate import st_getenv, st_apikey
from utils.st_retrieval import load_unique_ids, load_interview_plan

INTERVIEWERS = {
    "Humble": "./img/XriI.gif",
    "Relaxed": "./img/mask.gif",
    "Nervous": "./img/kerry.gif",
    "Friendly": "./img/mayor.gif",
}


def _log_interview(messages, plan_id, user_id=None):
    id = str(uuid.uuid4()) if user_id is None else user_id
    text = [f"[{plan_id}] Interview summary:", ]

    for m in messages:
        role = m["role"]
        content = m["content"]
        if role != "system":
            role = "Candidate" if role == "user" else "Interviewer"
            txt = f"{role}: {content}"
            text.append(txt)

    text = [t.replace("\n", " ") for t in text]
    text = "\n".join(text)
    with open(f"./db/interview_{id}.txt", "w") as f:
        f.write(text)

    return text


def reset_conversation():
    try:
        # # delete key
        # del st.session_state["api_key"]
        # del os.environ["OPENAI_API_KEY"]
        # logger.warning(f"Key deleted: {st_getenv('api_key', None)}. ")
        logger.warning(f"Environment variable deleted: {os.getenv('OPENAI_API_KEY')}.")
        streamlit_js_eval(js_expressions="parent.window.location.reload()")

    except Exception as e:
        logger.error(f"Error: {e}")
        st.error("Something went wrong. Please reload the page.")


def evaluate_user_answer(user_answer, api_key):
    try:
        question = st.session_state.messages[-1]["content"]
        prompt_task = "Evaluate candidate's answer (score: 1-10). " \
                      "Provide an example of the good answer (1-2 sentences):"
        prompt_data = f"Question: {question}\nAnswer: {user_answer}\n\n"
        prompt = f"{prompt_task}\n\n{prompt_data}"
        return openai_response_for_text(prompt, api_key)
    except Exception as e:
        logger.error(f"Error: {e}")
        return "Evaluation failed. Please try again."


def _split_text(text, max_len=400):
    if len(text) <= max_len:
        return [text, ]
    else:
        chunks = []
        while len(text) > max_len:
            idx = text[:max_len].rfind(".")
            chunks.append(text[:idx])
            text = text[idx:]
        chunks.append(text)
        return chunks


def st_init_chatbot():
    """
    This function initializes chatbot
    """
    with st.expander("Config", expanded=True):
        col1, col2 = st.columns((1, 1))
        with col1:
            st.markdown("#### ▶️ Select your Interviewer")
            interviewer = st.selectbox("Interviewer", list(INTERVIEWERS.keys()), label_visibility="collapsed")
            interviewer_name = st.text_input("Interviewer name", value="Ted", label_visibility="collapsed")
            st.session_state['assistant_avatar'] = INTERVIEWERS[interviewer]

            situation = "You are an Interviewer. You are conducting a job interview. " \
                        "Step by step ask me questions from the list below. After receiving the answer, " \
                        "write down the short comment and ask the next question. "
            messages = [{"role": "system", "content": situation}]

        with col2:
            st.image(INTERVIEWERS[interviewer], use_column_width=True)

        ids = ['1', ] + load_unique_ids()
        st.markdown("#### ▶️ Select your Interview Plan")
        plan_id = st.selectbox("Interview Plan", ids, label_visibility="collapsed")
        questions, plan_type = load_interview_plan(plan_id=plan_id)
        ice_breaker = f"Hi, I'm your interviewer. " \
                      f"My name is {interviewer_name}. What is your name?"

        prompt_task = f"After finishing the interview and providing the summary, write: 'Have a great day!'."
        questions = "\n".join([f"{idx + 1}. {q}" for idx, q in enumerate(questions)])
        content = f"{prompt_task}\n\nQUESTIONS:\n{questions}"
        messages_for_bot_init = [{"role": "system", "content": content},
                                 {"role": "assistant", "content": ice_breaker}, ]
        messages += messages_for_bot_init
        st.write(questions)

        return {"messages": messages, "plan_id": plan_id, "plan_type": plan_type, "interviewer": interviewer}


def main(admin=None):
    """
    This function is a main program function
    :return: None
    """
    st_apikey()
    api_key = st_getenv("api_key")

    if st.sidebar.button("Reset"):
        reset_conversation()

    col1, col2 = st.columns((4, 1))

    with col2:
        pass

    with col1:

        if "messages" not in st.session_state:

            config_data = st_init_chatbot()

            submitted = st.button("Submit config")

            if submitted and st_getenv('api_key') is not None:
                st.session_state["messages"] = config_data["messages"]
                st.session_state["plan_id"] = config_data["plan_id"]
                st.session_state["plan_type"] = config_data["plan_type"]
                st.experimental_rerun()
                st.success("Config submitted")
            elif submitted:
                st.markdown("⚠️ Please enter in the field above your OpenAI API key to continue.")

        else:
            # user_chat = st.chat_message("user")
            assistant_av = st.session_state['assistant_avatar']

            for msg in st.session_state.messages[:]:
                if msg["role"] == "user" and len(msg["content"]) > 0:
                    st.chat_message("user").write(msg["content"], )
                elif msg["role"] == "assistant":
                    st.chat_message("assistant", avatar=assistant_av).write(msg["content"])
                else:
                    logger.info("System message updated")

    if prompt := st.chat_input():
        # async def evaluate_user_answer(user_answer, api_key):
        with st.sidebar.expander("System message", expanded=False):
            evaluation = evaluate_user_answer(prompt, api_key)
            st.write("\n".join(_split_text(evaluation, max_len=400)))

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        msg = openai_response(st.session_state.messages, api_key, with_content=True)
        st.session_state.messages.append(msg)
        st.chat_message("assistant", avatar=assistant_av).write(msg.content)

        if "Have a great day" in msg.content:
            st.success("Interview completed")

            plan_id = st_getenv("plan_id", '1')
            logs = _log_interview(st.session_state.messages, plan_id=plan_id)
            st.session_state["logs"] = logs
            st.stop()
            feedback = streamlit_feedback(feedback_type="thumbs")

    if st.button("Evaluate"):
        st.success("Interview completed")
        messages_all = st_getenv("messages", [])

        if len(messages_all) <= 3:
            st.error("Please start the interview first.")
            st.stop()
        elif len(messages_all) <= 8:
            st.error("Not enough messages. Please answer at least 2 questions.")
            st.stop()

        plan_id = st_getenv("plan_id", '1')
        logs = _log_interview(st.session_state.messages, plan_id=plan_id)

        # get summary
        if st_getenv("plan_type") == "screening":
            assesment_instructions = screening_assessment + scr_assessment_format
        elif st_getenv("plan_type") == "tech":
            assesment_instructions = tech_assessment + tech_assessment_format
        else:
            assesment_instructions = assesment_default

        assesment_instructions = assesment_instructions.replace("{text}", logs)
        messages = [{"role": "system", "content": assesment_instructions}, ]

        with st.spinner("Generating summary..."):
            summary = openai_response(messages, api_key, with_content=False)
        st.write(summary)
        st.session_state["logs"] = logs
        st.stop()
        feedback = streamlit_feedback(feedback_type="thumbs")
        st.session_state["feedback"] = feedback

    if 'feedback' in st.session_state:
        plan = st_getenv("plan_id", '1')
        with open(f"./db/feedback_{plan}.txt", "a") as f:
            f.write(st.session_state["feedback"])


if __name__ == "__main__":
    main(admin=False)
