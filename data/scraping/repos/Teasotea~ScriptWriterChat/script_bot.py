import os

from langchain import PromptTemplate
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.chains.summarize import load_summarize_chain

from langchain.text_splitter import CharacterTextSplitter
import textwrap

from dotenv import load_dotenv
import streamlit as st
from streamlit_chat import message

from prompt_lib import (
    STARTER_PROMPT,
    STORY_YT_PROMPT,
    LIST_YT_PROMPT,
    STORY_SC_PROMPT,
    LIST_SC_PROMPT,
)

SOCIAL_MEDIA = ["YouTube", "SnapChat"]
SCRIPT_TYPE = ["Story Script", "List Script"]


def init():
    load_dotenv()
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")
    st.set_page_config(page_title="ScriptWriter Bot", page_icon="🖋🎥")


def main():
    init()
    chat = ChatOpenAI(temperature=0.1, model_name="gpt-3.5-turbo")
    if "messages" not in st.session_state:
        st.session_state.messages = [SystemMessage(input_variables=["input", "history"], content=STARTER_PROMPT)]

    st.header("ScriptWriter Bot 🖋🎥")
    PROMPT = PromptTemplate(template=STARTER_PROMPT, input_variables=["input", "history"])
    conversation = ConversationChain(
        prompt=PROMPT, llm=chat, memory=ConversationSummaryBufferMemory(llm=chat, max_token_limit=650), verbose=True
    )

    print(conversation.prompt.template)  #

    def generate_response(prompt):
        return conversation.predict(input=prompt)

    with st.sidebar:
        st.header("🖋 Script Details")
        st.subheader("Specify below the topic, type of script, and social media platform: ")
        user_input = st.text_input("Enter the topic: ", key="user_input")
        script_type = st.selectbox("Select the type of script", [None, SCRIPT_TYPE[0], SCRIPT_TYPE[1]])

        # send to list utils:
        if script_type == SCRIPT_TYPE[1]:
            topX = st.text_input("Enter number of items in the list (Top X..). 5-10 is recommended", key="top X")
            if topX:
                topX = int(topX)
        ####################

        platform = st.selectbox("Select platform", [None, SOCIAL_MEDIA[0], SOCIAL_MEDIA[1]])

        load_text = st.text_area("Text to extend knowledge (optional): ", key="load_text")

        is_generate_script = st.button("Generate script")

        # Improve this logic:
        if is_generate_script and not user_input:
            st.error('Click the "Generate script" button to generate script.')
        ####################

        if "generated" not in st.session_state:
            start_writer = "Hi! I am your ScriptWriter Bot! Let's get started!"
            st.session_state["generated"] = [start_writer]

        if "past" not in st.session_state:
            st.session_state["past"] = ["Hi!"]

        # load_link = st.text_input("Link on new article to extend knowledge: ", key="load_link")

        # st.write("Click the button to save the script to Google Docs.")
        # is_save_script = st.button("Save script to Google Docs")
        # if is_save_script and is_generate_script:
        #     st.write("Script will be uploaded.")
        # elif is_save_script and not is_generate_script:
        #     st.error("Nothing to save. Please, enter the topic and click the button to generate script.")

    if user_input and script_type and platform:
        if script_type == SCRIPT_TYPE[0] and platform == SOCIAL_MEDIA[0]:
            PROMPT_ORDER = STORY_YT_PROMPT(user_input)
        elif script_type == SCRIPT_TYPE[0] and platform == SOCIAL_MEDIA[1]:
            PROMPT_ORDER = STORY_SC_PROMPT
        elif script_type == SCRIPT_TYPE[1] and platform == SOCIAL_MEDIA[0]:
            PROMPT_ORDER = LIST_YT_PROMPT(topX, user_input)
        elif script_type == SCRIPT_TYPE[1] and platform == SOCIAL_MEDIA[1]:
            PROMPT_ORDER = LIST_SC_PROMPT(topX, user_input)

        # link_context = "use context from html page in script: " + str(get_links(load_link)) if load_link else ""
        if load_text:
            chain = load_summarize_chain(chat, chain_type="map_reduce", verbose=True)
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = text_splitter.create_documents([load_text])
            output_summary = chain.run(docs)
            wrapped_text = textwrap.fill(output_summary, width=300)

        additional_text = "this text is ground truth, use it in script: " + wrapped_text if load_text else ""

        list_or_content = ""
        if script_type and platform and is_generate_script:
            st.session_state.messages.append(HumanMessage(content="topic:" + user_input))
            st.session_state.past.append(user_input)
            for i, prompt in enumerate(PROMPT_ORDER):
                with st.spinner("Please, wait. I am in the process of creating script for you..."):
                    if script_type == SCRIPT_TYPE[1]:
                        final_prompt = (
                            "SESSION INFO: topic - Top "
                            + ", items are: "
                            + str(topX)
                            + " "
                            + user_input
                            + "; "
                            + list_or_content
                            + " "
                            + additional_text
                            + "; "
                            + prompt
                        )
                    else:
                        final_prompt = (
                            "SESSION INFO: topic - " + user_input + "; " + list_or_content + " " + additional_text + "; " + prompt
                        )
                    response = generate_response(final_prompt)

                    print("RESPONSE #" + str(i + 1) + ": " + response)
                    if i == 1:
                        list_or_content = (
                            "Use items from this list for further generation! Don't tell a story of same item multiple times" + response
                        )
                    st.session_state.generated.append(response)  # "RESPONSE #" + str(i + 1) + ": " + response)
                    st.session_state.past.append("Continue")

        elif not is_generate_script:
            st.error('Click the "Generate script" button to generate script.')

    if st.session_state["generated"]:
        message(st.session_state["past"][0], is_user=True, key="_user")
        message(st.session_state["generated"][0], key=str(0))

        final_message_past, final_message_gen = [], []
        for i in range(1, len(st.session_state["generated"])):
            if i != 2 or (platform == SOCIAL_MEDIA[1] and script_type == SCRIPT_TYPE[0]):
                final_message_past.append(st.session_state["past"][i])
                final_message_gen.append(st.session_state["generated"][i])
        # message(' '.join(final_message_past), key="_user")
        message(user_input, is_user=True, key="_user2")
        message(" ".join(final_message_gen), key="generated")


if __name__ == "__main__":
    main()
