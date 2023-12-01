import streamlit as st
import openai
import re
from gradio_client import Client

# This function checks if the content is a valid C# method
def is_valid_csharp_method(content):
    pattern = r"(public|private|internal|protected)(\s+static)?\s+\w+\s+\w+\("
    return re.search(pattern, content) is not None

# Modularized function to interact with an API (in this case, ChatGPT)
def send_to_api(content, endpoint, test_framework):
    if endpoint == "gpt-3.5-turbo":
        prompt = f"Please provide {test_framework} unit tests for the following C# method to maximize code coverage. Only return the code, no additional explanations:\n{content}"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=3000
        )
        return response['choices'][0]['message']['content']
    elif endpoint == "mosaicml/mpt-7b-instruct":
        prompt = f"Please provide {test_framework} unit tests for the following C# method to maximize code coverage. Only return the code, no additional explanations:\n{content}"
        
        client = Client("https://faefb20476a7197ca8.gradio.live")
        params = '{"max_new_tokens": 100, "temperature": 0.05}'  # Use a single dictionary
        response = client.predict(prompt, params, api_name="/greet")
        return response
        
        
def explain_code(content, endpoint):
    """Function to send code to the API for explanation."""
    prompt = f"Please explain the following C# method:\n\n{content}. Only return the explanation, no code."
    if endpoint == "gpt-3.5-turbo":
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        return response['choices'][0]['message']['content']
    elif endpoint == "mosaicml/mpt-7b-instruct":
        client = Client("https://faefb20476a7197ca8.gradio.live")
        params = '{"max_new_tokens": 100, "temperature": 0.05}'
        response = client.predict(prompt, params, api_name="/greet")
        return response
        

def critique_input_method(content):
    prompt = f"Please critique the following C# method and provide suggestions for improvement:\n\n{content}. Response should only contain the critique and suggestions."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt},
            {"role": "system", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )
    return response['choices'][0]['message']['content']
    
def translate_code(content, target_language):
    """Function to translate code to another language using LLM."""
    prompt = f"Translate the following C# code to {target_language}:\n\n{content}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt},
            {"role": "system", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )
    return response['choices'][0]['message']['content']
    
def chat_with_code(content, user_query):
    """Function to chat with the code using LLM."""
    prompt = f"Code:\n{content}\n\nUser: {user_query}\nCode: "
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1000
    )
    return response['choices'][0]['message']['content']

import streamlit as st
import openai
import re
from gradio_client import Client

# This function checks if the content is a valid C# method
def is_valid_csharp_method(content):
    pattern = r"(public|private|internal|protected)(\s+static)?\s+\w+\s+\w+\("
    return re.search(pattern, content) is not None

# Modularized function to interact with an API (in this case, ChatGPT)
def send_to_api(content, endpoint, test_framework):
    if endpoint == "gpt-3.5-turbo":
        prompt = f"Please provide {test_framework} unit tests for the following C# method to maximize code coverage. Only return the code, no additional explanations:\n{content}"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=3000
        )
        return response['choices'][0]['message']['content']
    elif endpoint == "mosaicml/mpt-7b-instruct":
        prompt = f"Please provide {test_framework} unit tests for the following C# method to maximize code coverage. Only return the code, no additional explanations:\n{content}"
        
        client = Client("https://faefb20476a7197ca8.gradio.live")
        params = '{"max_new_tokens": 100, "temperature": 0.05}'  # Use a single dictionary
        response = client.predict(prompt, params, api_name="/greet")
        return response
        
        
def explain_code(content, endpoint):
    """Function to send code to the API for explanation."""
    prompt = f"Please explain the following C# method:\n\n{content}. Only return the explanation, no code."
    if endpoint == "gpt-3.5-turbo":
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        return response['choices'][0]['message']['content']
    elif endpoint == "mosaicml/mpt-7b-instruct":
        client = Client("https://faefb20476a7197ca8.gradio.live")
        params = '{"max_new_tokens": 100, "temperature": 0.05}'
        response = client.predict(prompt, params, api_name="/greet")
        return response
        

def critique_input_method(content):
    prompt = f"Please critique the following C# method and provide suggestions for improvement:\n\n{content}. Response should only contain the critique and suggestions."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt},
            {"role": "system", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )
    return response['choices'][0]['message']['content']
    
def translate_code(content, target_language):
    """Function to translate code to another language using LLM."""
    prompt = f"Translate the following C# code to {target_language}:\n\n{content}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt},
            {"role": "system", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )
    return response['choices'][0]['message']['content']
    
def chat_with_code(content, user_query):
    """Function to chat with the code using LLM."""
    prompt = f"Code:\n{content}\n\nUser: {user_query}\nCode: "
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1000
    )
    return response['choices'][0]['message']['content']

def main():


    # Display the centered logo
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("imgs/tioca_chal_coin.png", width=300)  # Adjust width as needed

    st.title("Unit Test Generation Tool")
    # Organize "Select API Endpoint", "Select Test Framework", and "Select Target Language" in the same row
    col1, col2, col3 = st.columns(3)
    with col1:
        api_endpoint = st.selectbox("Select API Endpoint", ["mosaicml/mpt-7b-instruct", "gpt-3.5-turbo"])
    
    # Conditionally display the OpenAI API key input based on the selected API endpoint
    if api_endpoint == "gpt-3.5-turbo":
        openai_key = st.text_input("Enter OpenAI API Key:", type="password")
        if openai_key:
            openai.api_key = openai_key

    with col2:
        test_framework = st.selectbox("Select Test Framework", ["NUnit", "xUnit.net", "MSTest"])
    
    with col3:
        target_language = st.selectbox("Select Target Language for Translation", ["Python", "Java", "JavaScript", "C++"])

    # Upload C# method
    uploaded_file = st.file_uploader("Upload C# method (.cs file)", type=["cs"])

    if uploaded_file:
        content = uploaded_file.read().decode()
        st.markdown("---")
        st.subheader("Uploaded C# Method")
        st.code(content, language='csharp')

        if is_valid_csharp_method(content):
            # Organize the buttons side by side
            btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
            with btn_col1:
                if st.button("Generate Unit Tests"):
                    with st.spinner('Generating Unit Tests...'):
                        st.session_state.unit_tests = send_to_api(content, api_endpoint, test_framework)
                    with open("generated_tests.cs", "w") as f:
                        f.write(st.session_state.unit_tests)

            with btn_col2:
                if st.button("Critique Input Method"):
                    with st.spinner('Critiquing Method...'):
                        st.session_state.critique = critique_input_method(content)

            with btn_col3:
                if st.button("Explain the Code"):
                    with st.spinner('Explaining the Code...'):
                        st.session_state.explanation = explain_code(content, api_endpoint)

            with btn_col4:
                if st.button("Translate Code"):
                    with st.spinner('Translating the Code...'):
                        st.session_state.translated_code = translate_code(content, target_language)

            # Display results
            if "unit_tests" in st.session_state:
                st.markdown("---")
                st.subheader(f"Generated {test_framework} Tests")
                st.code(st.session_state.unit_tests, language='csharp')
                downloaded_file = open("generated_tests.cs", "r").read()
                st.download_button("Download Generated Tests", data=downloaded_file, file_name="generated_tests.cs", mime="text/plain")

            if "critique" in st.session_state:
                st.markdown("---")
                st.subheader("Critique of Input Method")
                st.text_area("Critique", st.session_state.critique, height=150)

            if "explanation" in st.session_state:
                st.markdown("---")
                st.subheader("Explanation of Input Method")
                st.text_area("Explanation", st.session_state.explanation, height=150)

            if "translated_code" in st.session_state:
                st.markdown("---")
                st.subheader(f"Translated Code ({target_language})")
                st.code(st.session_state.translated_code, language=target_language.lower())

            # Chat with code section
            st.markdown("---")
            st.subheader("Chat with the Code")
            user_query = st.text_input("Ask a question about the code:", key="chat_input")

            if user_query and not "last_query" in st.session_state:
                st.session_state.last_query = user_query
                with st.spinner("Waiting for response..."):
                    response = chat_with_code(content, user_query)
                    st.session_state.chat_response = response
                st.code(st.session_state.chat_response, language='csharp')
            elif "last_query" in st.session_state and st.session_state.last_query != user_query:
                # Reset last_query to allow new queries
                del st.session_state.last_query

        else:
            st.error("The uploaded file does not contain a valid C# method.")

if __name__ == "__main__":
    main()



