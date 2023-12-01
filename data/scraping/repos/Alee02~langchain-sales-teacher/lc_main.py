import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langsmith import Client
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from get_prompt import load_prompt, load_prompt_with_questions

st.set_page_config(page_title="Advanced Sales Process: Training", initial_sidebar_state="collapsed")

st.markdown(
    """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
</style>
""",
    unsafe_allow_html=True,
)
st.title("🤝 Advanced Sales Process: Training")
button_css = """.stButton>button {
    color: #4F8BF9;
    border-radius: 50%;
    height: 2em;
    width: 2em;
    font-size: 4px;
}"""
st.markdown(f'<style>{button_css}</style>', unsafe_allow_html=True)

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

# Lesson selection dictionary
lesson_guides = {
    "Lesson 1: Introduction to Sales Fundamentals": {
        "file": "sales_guides/introduction_guide.txt",
        "description": "This lesson introduces the core concepts of selling and the importance of understanding customer needs."
    },
    "Lesson 2: Prospecting and Lead Generation": {
        "file": "sales_guides/prospecting_guide.txt",
        "description": "Learn the strategies to find and qualify potential clients. Understand the importance of targeted lead generation."
    },
    "Lesson 3: Crafting the Perfect Pitch": {
        "file": "sales_guides/pitching_guide.txt",
        "description": "This lesson covers the art of crafting a persuasive sales pitch and adapting to different client needs."
    },
    "Lesson 4: Handling Objections": {
        "file": "sales_guides/objections_guide.txt",
        "description": "Overcome common sales objections using tested techniques and methodologies."
    },
    "Lesson 5: Closing Techniques": {
        "file": "sales_guides/closing_guide.txt",
        "description": "Master the skills to seal the deal. Understand when and how to ask for the sale."
    },
    "Lesson 6: Building Long-Term Relationships": {
        "file": "sales_guides/relationships_guide.txt",
        "description": "Learn the importance of post-sale service and how to foster long-term client relationships."
    },
    "Lesson 7: Leveraging Technology in Sales": {
        "file": "sales_guides/technology_guide.txt",
        "description": "Understand how to use modern tools and platforms to enhance your sales process and stay ahead of the competition."
    }
}

# Initialize LangSmith client
client = Client()

# Lesson selection sidebar
lesson_selection = st.sidebar.selectbox("Select Lesson", list(lesson_guides.keys()))

# Display lesson content and description based on selection
lesson_info = lesson_guides[lesson_selection]
lesson_content = open(lesson_info["file"], "r").read()
lesson_description = lesson_info["description"]

# Radio buttons for lesson type selection
lesson_type = st.sidebar.radio("Select Lesson Type", ["Instructions based lesson", "Interactive lesson with questions"])

# Clear chat session if dropdown option or radio button changes
if st.session_state.get("current_lesson") != lesson_selection or st.session_state.get("current_lesson_type") != lesson_type:
    st.session_state["current_lesson"] = lesson_selection
    st.session_state["current_lesson_type"] = lesson_type
    st.session_state["messages"] = [AIMessage(content="Welcome! This short course will help you get started with Sales. Let me know when you're all set to jump in!")]

# Display lesson name and description
st.markdown(f"**{lesson_selection}**")
st.write(lesson_description)

# Message handling and interaction
def send_feedback(run_id, score):
    client.create_feedback(run_id, "user_score", score=score)

for msg in st.session_state["messages"]:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    else:
        st.chat_message("assistant").write(msg.content)

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        model = ChatOpenAI(streaming=True, callbacks=[stream_handler], model="gpt-3.5-turbo-16k")

        if lesson_type == "Instructions based lesson":
            prompt_template = load_prompt(content=lesson_content)
        else:
            prompt_template = load_prompt_with_questions(content=lesson_content)

        chain = LLMChain(prompt=prompt_template, llm=model)

        response = chain(
            {"input": prompt, "chat_history": st.session_state.messages[-20:]},
            include_run_info=True,
            tags=[lesson_selection, lesson_type]
        )
        st.session_state.messages.append(HumanMessage(content=prompt))
        st.session_state.messages.append(AIMessage(content=response[chain.output_key]))
        run_id = response["__run"].run_id

        col_blank, col_text, col1, col2 = st.columns([10, 2, 1, 1])
        with col_text:
            st.text("Feedback:")

        with col1:
            st.button("👍", on_click=send_feedback, args=(run_id, 1))

        with col2:
            st.button("👎", on_click=send_feedback, args=(run_id, 0))
