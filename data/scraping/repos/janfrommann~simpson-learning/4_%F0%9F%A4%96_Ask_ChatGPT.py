import openai
import streamlit as st

# Sidebar for API key input
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")

# Set the OpenAI API key
if openai_api_key:
    openai.api_key = openai_api_key

# Main page layout
st.title("💬 Talk to ChatGPT")
st.caption("This is a Chatbot based on OpenAI ChatGPT")

# Initialize messages if they don't exist in the session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Display the chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Initialize 'user_input' to None to avoid NameError
user_input = None  # crucial; it initializes 'user_input'

# Sidebar for predefined prompts and responses
with st.sidebar:
    st.header("Predefined Prompts")
    prompt_buttons = {
        "Explain Confounders": "Please explain what confounders are, why they can be hidden, and how they relate to Simpsons Paradox. Tailor your explaination to smart high-school students",
        "Tell me about Spurious Correlations": "Please explain what Spurious Correlations are and how they relate to Simpsons Paradox and hidden confounders. Tailor your explaination to smart high-school students",
        "How can I recognise Simpson's Paradox?": "Please explain how you can recognise Simpsons Paradox in a given scenario. What should I watch out for? Tailor your explaination to smart high-school students",
        "What does aggregate and disaggreate view mean?": "Explain the differnces between the aggregate and disaggreate view in the context of Simpsons Paradox. Tailor your explaination to smart high-school students",
        "Please define causality": "Explain the meaning of causality in the context of Simpsons Paradox. Tailor your explaination to smart high-school students",
        "How do I decide if I should trust the otal or subgroup view": "In a case of Simpsons Paradox, how do you go about deciding weather the aggregate or disaggregate view is a better representation of the truth? Tailor your explaination to smart high-school students"

    }

    # Loop over the predefined prompts
    for prompt in prompt_buttons:
        if st.button(prompt):
            user_input = prompt_buttons[prompt]  # new value if button is pressed

    st.header("Responses")
    response_buttons = {
        "Give more details": "Can you provide more details? If possible, give specific examples",
        "Simplify your answer": "That seems complicated. Can you explain it in simpler terms?"
    }

    # Loop over the responses
    for response in response_buttons:
        if st.button(response):
            user_input = response_buttons[response]  # Assigns new value if button is pressed

if user_input:  # This check is safe now as 'user_input' is defined in the scope
    # Add the user's message to the chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Get a response from the GPT-3 model
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    msg = response.choices[0].message
    st.session_state.messages.append(msg)
    st.chat_message("assistant").write(msg.content)
