import streamlit as st
import openai

# Function to query OpenAI ChatGPT
def query_chatgpt(prompt, config="config1"):
    # You can have different configurations by adjusting parameters like model, temperature, max tokens etc.
    if config == "config1":
        response = openai.Completion.create(engine="davinci-codex", prompt=prompt, max_tokens=100)
    elif config == "config2":
        response = openai.Completion.create(engine="davinci-codex", prompt=prompt, max_tokens=50, temperature=0.7)
    else:
        response = openai.Completion.create(engine="davinci", prompt=prompt, max_tokens=30, temperature=0.9)
    return response.choices[0].text


# Main function
def main():
    # Create the panel
    st.title("ChatGPT Panel")

    # Initialize session state variables
    if 'prompt' not in st.session_state:
        st.session_state['prompt'] = "Tell me a joke."
    if 'config' not in st.session_state:
        st.session_state['config'] = "config1"

    # Refresh button
    if st.button("Refresh"):
        answer = "refreshed......" #query_chatgpt(st.session_state['prompt'], st.session_state['config'])
        st.write(answer)

    # Settings button
    if st.button("Settings"):
        with st.sidebar:
            # Input for the prompt
            prompt = st.text_input("Enter the prompt", st.session_state['prompt'])
            st.session_state['prompt'] = prompt

            # Dropdown for output type (configurations)
            config = st.selectbox("Select output configuration", ["config1", "config2", "config3"])
            st.session_state['config'] = config

    # Display response
    answer = "answered............" #query_chatgpt(st.session_state['prompt'], st.session_state['config'])
    st.write(answer)


if __name__ == "__main__":
    main()
