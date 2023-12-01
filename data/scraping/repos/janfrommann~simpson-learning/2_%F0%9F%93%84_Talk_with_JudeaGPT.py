import streamlit as st
import anthropic

with st.sidebar:
    anthropic_api_key = st.text_input("Anthropic API Key", key="file_qa_api_key", type="password")
    "If you don't have an API Key, you can go directly to [Claude](https://claude.ai). Just make sure to let Claude know that you are a highschool student."


st.title("📝 Ask 🤖Judea")
uploaded_file = st.file_uploader("Upload the paper from the course material", type=("txt", "md"))
question = st.text_input(
    "Ask something about the paper 'Understanding Simpson’s Paradox' by Judea Pearl",
    placeholder="Can you give me a short summary?",
    disabled=not uploaded_file,
)

if uploaded_file and question and not anthropic_api_key:
    st.info("Please add your Anthropic API key to continue.")

if uploaded_file and question and anthropic_api_key:
    article = uploaded_file.read().decode()
    prompt = f"""{anthropic.HUMAN_PROMPT} Here's an article:\n\n<article>
    {article}\n\n</article>\n\n{"Answer the following question for highschool students:" + question}{anthropic.AI_PROMPT}"""

    client = anthropic.Client(api_key=anthropic_api_key)
    response = client.completions.create(
        prompt=prompt,
        stop_sequences=[anthropic.HUMAN_PROMPT],
        model="claude-v1", #"claude-2" for Claude 2 model
        max_tokens_to_sample=100,
    )
    st.write("### Answer")
    st.write(response.completion)
