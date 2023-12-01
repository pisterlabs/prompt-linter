#Importing required packages
import streamlit as st
import promptlayer
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import uuid
import math

INIT_PROMPT = """
\n\nHuman: You are MapMentor a trainer in Wardley Mapping. You will help the users learn about Wardley Mapping
Here are some important rules for the interaction:
- Always stay in character, as MapMentor a Wardley Mapping trainer.  
- If you are unsure how to respond, respond with another question.
- Always use a liberationism pedagogy training approach.
- Remember to state which module of the course we are currently learning.
"""

TRAINING_PROMPT = """
Here is an outline for a training course that you will give to the user. It covers the key principles of Wardley Mapping:

Module 1 - Introduction to Wardley Mapping

Purpose and benefits of mapping
Understanding value chains and situational awareness
Overview of doctrine and foundational concepts
Module 2 - Structure of Wardley Maps

Components, activities, and the value chain
Evolution axis and commodity forms
Anchors, chains, and dependencies
Module 3 - Developing Wardley Maps

Gathering insight on activities, capabilities, and needs
Positioning and classifying map elements
Adding annotations and context
Module 4 - Using Maps for Decision Making

Identifying structural vs situational change
Applying doctrine to strategic planning
Mapping out competing value chains
Developing actionable insights from maps
Module 5 - Advanced Concepts

Ecosystem models and community maps
Climate patterns and their impact
Mapping organizational culture
Handling uncertainty and unknowns
Module 6 - Facilitating Wardley Mapping

Workshops for collaborative mapping
Engaging leadership and stakeholders
Promoting adoption and managing skeptics
For each module, we would provide concepts, examples, hands-on exercises, and practice activities to build skills.
Please let me know if you would like me to expand on any part of this high-level curriculum outline for a Wardley Mapping training course.
I'm happy to provide more details on how to effectively teach this methodology.
"""

INTRO_PROMPT = """
Hello, I'm MapMentor  \n
This course is designed as an interactive learning experience to build your skills in Wardley Mapping from the ground up. We will cover the key principles, components, and steps for creating powerful maps.  \n

The course is organized into 6 modules:  \n
Module 1 provides an introduction to the purpose, benefits, and foundational concepts of mapping. We discuss how it helps with situational awareness and strategic planning.  \n
Module 2 focuses on the structure of Wardley Maps - the components like activities, evolution axis, dependencies. You'll learn how to visualize your value chain.  \n
Module 3 is all about developing maps hands-on. We'll practice gathering insights, positioning elements, adding annotations to create meaningful maps.  \n
Module 4 shifts to using completed maps for strategic analysis and decision making. You'll apply doctrine to interpret maps and generate insights.  \n
Module 5 covers more advanced concepts like mapping ecosystems, organizational culture, and handling uncertainty.  \n
Finally Module 6 is on facilitating mapping workshops and driving adoption.  \n
Each module includes concepts, examples, exercises, and practice activities to build your skills. You'll have opportunities to create maps, iterate on them, and apply them to scenario-based challenges.  \n
I'm looking forward to exploring all aspects of Wardley Mapping with you in this course! Please let me know if you would like me to elaborate on any part of the curriculum.
"""

REG_PROMPT = """
\n\nHuman: Here is the user's question about Wardley Mapping:
<question>
{QUESTION}
</question>
\n\nAssistant: [MapMentor] <response>
"""

# Anthropic Claude pricing: https://cdn2.assets-servd.host/anthropic-website/production/images/model_pricing_may2023.pdf
PRICE_PROMPT = 1.102E-5
PRICE_COMPLETION = 3.268E-5

#MODEL = "claude-1"
MODEL = "claude-2"
#MODEL = "claude-v1-100k"

new_prompt = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    
st.set_page_config(page_title="Anthropic - ChatBot")
st.sidebar.title("Anthropic - ChatBot")
st.sidebar.title("Wardley Mapping Mentor")
st.sidebar.divider()
st.sidebar.markdown("Developed by Mark Craddock](https://twitter.com/mcraddock)", unsafe_allow_html=True)
st.sidebar.markdown("Current Version: 0.0.2")
st.sidebar.markdown("Using claude-2 API")
st.sidebar.markdown(st.session_state.session_id)
st.sidebar.divider()

# Check if the user has provided an API key, otherwise default to the secret
user_claude_api_key = st.sidebar.text_input("Enter your Anthropic API Key:", placeholder="sk-...", type="password")

st.sidebar.markdown("Tokens")
total_tokens = st.sidebar.empty()

if "claude_model" not in st.session_state:
    st.session_state["claude_model"] = MODEL

if "messages" not in st.session_state:
    st.session_state["messages"] = []
    st.session_state.messages.append({"role": "assistant", "content": INTRO_PROMPT})

if "all_prompts" not in st.session_state:
    st.session_state["all_prompts"] = INIT_PROMPT + TRAINING_PROMPT

def count_used_tokens(prompt, completion):
    prompt_token_count = client.count_tokens(prompt)
    completion_token_count = client.count_tokens(completion)
    prompt_cost = prompt_token_count * PRICE_PROMPT
    completion_cost = completion_token_count * PRICE_COMPLETION
    total_cost = prompt_cost + completion_cost
    total_cost = math.ceil(total_cost * 100) / 100
    return (
        prompt_token_count,
        completion_token_count,
        total_cost
    )
    
if user_claude_api_key:
    # If the user has provided an API key, use it
    # Swap out Anthropic for promptlayer
    promptlayer.api_key = st.secrets["PROMPTLAYER"]
    anthropic = promptlayer.anthropic
    client=anthropic.Anthropic(
      # defaults to os.environ.get("ANTHROPIC_API_KEY")
      api_key=user_claude_api_key,
    )
else:
    st.warning("Please enter your Anthropic Claude API key", icon="⚠️")

for message in st.session_state.messages:
    if message["role"] in ["user", "assistant"]:
        with st.chat_message(message["role"]):
            new_prompt.append(message["content"])
            st.markdown(message["content"])
            
if user_claude_api_key:
    if user_input := st.chat_input("How can I help with Wardley Mapping?"):
        prompt = REG_PROMPT.format(QUESTION = user_input)
        st.session_state.all_prompts += prompt
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
        full_response = ""
        try:
            for response in client.completions.create(
                prompt=st.session_state.all_prompts,
                stop_sequences=["</response>"],
                model=MODEL,
                max_tokens_to_sample=500,
                stream=True,
                pl_tags=["anthropic-chatbot", st.session_state.session_id]
            ):
                full_response += response.completion
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        except anthropic.APIConnectionError as e:
            st.error("The server could not be reached")
            print(e.__cause__)  # an underlying Exception, likely raised within httpx.
        except anthropic.RateLimitError as e:
            st.error("A 429 status code was received; we should back off a bit.")
        except anthropic.APIStatusError as e:
            st.error("Another non-200-range status code was received")
            st.error(e.status_code)
            st.error(e.response)      
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.session_state.all_prompts += full_response
        prompt_token_count, completion_token_count, total_cost = count_used_tokens(prompt, full_response)
        total_tokens.markdown("Prompt: " + str(prompt_token_count) + "  \nCompletion: " + str(completion_token_count) + "  \nTotal Cost: $" + str(total_cost))
