# Import necessary libraries
import streamlit as st
from langchain.chains import ConversationChain
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI

def initialize_streamlit():
    """Set up Streamlit page configurations and title."""
    st.set_page_config(
        page_title="StudyGPT",
        page_icon=":mortar_board:",
        initial_sidebar_state = "collapsed"
    )
    st.title("🤖 AI Tutor")
    st.caption("✨ Your AI study partner, ask me anything!")

def initialize_session_states():
    """Initialize session states for entity_memory, generated, and past."""
    if 'entity_memory' not in st.session_state:
        st.session_state.entity_memory = ConversationBufferWindowMemory(k=10)

    if "generated" not in st.session_state:
        st.session_state["generated"] = []

    if "past" not in st.session_state:
        st.session_state["past"] = []

def load_prompt_template():
    """Load the prompt template from the text file."""
    with open("ai_tutor.txt", "r") as f:
        template = f.read()
    return template

def load_chain(api_key, template):
    """Initialize the ConversationChain object."""
    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=api_key,
        model_name="gpt-3.5-turbo",
        verbose=False
    )

    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=template
    )

    conversation = ConversationChain(
        llm=llm,
        prompt=prompt,
        memory=st.session_state.entity_memory,
    )

    return conversation

def get_response(conversation, input_text):
    """Get a response from the conversation object based on the input_text."""
    output = conversation.run(input=input_text)
    return output

def display_previous_messages():
    """Display previous messages in the chat session."""
    if st.session_state["past"]:
        for i in range(len(st.session_state["past"])):
            answer_container.markdown(f"""🤓 **YOU:** {st.session_state["past"][i]}""")
            if i < len(st.session_state["generated"]):
                answer_container.markdown(f"""🤖 **AI:** {st.session_state["generated"][i]}""")

def handle_form_submission(conversation):
    """Handle form submission and update the chat session with new messages."""
    if submitted and input_text:
        st.session_state.past.append(input_text)
        answer_container.markdown(f"""🤓 **YOU:** {input_text}""")

        with st.spinner("💭 Waiting for response..."):
            answer = get_response(conversation, input_text)
            answer_container.markdown(f"""🤖 **AI:** {answer}""")

        if answer:
            st.session_state.generated.append(answer)

# Main code
initialize_streamlit()
initialize_session_states()
api_key = st.secrets["OPENAI_API_KEY"]
template = load_prompt_template()
conversation = load_chain(api_key, template)
answer_container = st.container()
ask_form = st.empty()

display_previous_messages()

with ask_form.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([10, 1])
    input_text = col1.text_area(" ", max_chars=2000, key="input", placeholder="Type your question here...", label_visibility="collapsed",)
    submitted = col2.form_submit_button("💬")
    handle_form_submission(conversation)
