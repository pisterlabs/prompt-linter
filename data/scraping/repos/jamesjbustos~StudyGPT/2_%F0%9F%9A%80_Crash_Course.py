import openai
import streamlit as st
import re
import subprocess
import json
import os
import sys

# Streamlit page configurations and title
st.set_page_config(
    page_title="StudyGPT",
    page_icon=":mortar_board:",
    initial_sidebar_state = "collapsed"
)
st.title("🚀 Crash Course")
st.caption("✨ Your ultimate learning companion - choose any topic and accelerate your growth!")

# Load API Key
api_key = st.secrets["OPENAI_API_KEY"]

def load_content_from_json(topic, expected_sections_count):
    if not os.path.exists(f"{topic}_content.json"):
        return {}
    with open(f"{topic}_content.json", "r") as f:
        content_data = json.load(f)
    if len(content_data) != expected_sections_count:
        return {}
    return content_data

@st.cache_data(show_spinner=False)
def generate_outline(topic):
    prompt = [{"role": "system", "content": "You are a helpful assistant that creates an outline for learning a given topic."},
              {"role": "user", "content": f"""Create an outline for learning the topic: {topic}. Each topic should be on a new line, with a dash infront of each topic, nothing else. Generate a maximum amount of 5 topics for this outline. Please omit any headings with title such as Introduction to ...
               Example:
               
                - Topic 1
                - Topic 2
                - Topic 3
                - Topic 4
                - Topic 5
               
               """}]
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prompt
    )
    return completion.choices[0].message.content

@st.cache_data
def parse_content(content):
    sections = re.split(r'\n{1,}', content)
    return [section.strip("- ").strip() for section in sections]

user_topic = st.text_input("Enter a topic")

if user_topic:
    user_topic = user_topic.lower()
    with st.spinner('✨ Generating outline...'):
        outline = generate_outline(user_topic)
    sections = parse_content(outline)
    
    if sections:
        with st.spinner('🧙‍♂️ Generating course...'):
            # Generate content if not already available
            if not os.path.exists(f"{user_topic}_content.json"):
                subprocess.run([f"{sys.executable}", "generate_content.py", user_topic, *sections])
            
            content_data = load_content_from_json(user_topic, len(sections))

            selected_module = st.selectbox("Select a module:", sections)
            module_content = content_data.get(selected_module, "No content available for this module.")
            # st.markdown(f"## {selected_module}")
            st.markdown(module_content)
    else:
        st.write("No content available for given topic.")
