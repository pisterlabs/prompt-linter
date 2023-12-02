import streamlit as st
import base64
import requests
import json
import ssl
import os
import urllib.request
from io import BytesIO
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms.azureml_endpoint import AzureMLOnlineEndpoint, LlamaContentFormatter
from dotenv import load_dotenv

st.set_page_config(page_title='💬 Llama 2')


def image_to_base64(image):
    """
    Convert an image to its base64 representation.
    """

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context


allowSelfSignedHttps(True)  # this line is needed if you use self-signed certificate in your scoring service.


def askdocuments2(
        question):
    # Request data goes here
    # The example below assumes JSON formatting which may be updated
    # depending on the format your endpoint expects.
    # More information can be found here:
    # https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script
    formatter = LlamaContentFormatter()
    data = formatter.format_request_payload(prompt=question, model_kwargs={"temperature": 0.1, "max_tokens": 300})
    body = data
    load_dotenv()
    url = os.getenv("AZUREML_ENDPOINT_URL")
    # Replace this with the primary/secondary key or AMLToken for the endpoint
    api_key = os.getenv("AZUREML_ENDPOINT_API_KEY")
    if not api_key:
        raise Exception("A key should be provided to invoke the endpoint")

    # The azureml-model-deployment header will force the request to go to a specific deployment.
    # Remove this header to have the request observe the endpoint traffic rules
    headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + api_key), 'azureml-model-deployment': 'llama-2-7b-12-luis'}

    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)

        result = response.read()
        decoded_data = json.loads(result.decode('utf-8'))
        text = decoded_data[0]["0"]
        return text
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))

        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        print(error.info())
        print(error.read().decode("utf8", 'ignore'))


def askdocuments(
        text):

    try:
        content_formatter = LlamaContentFormatter()
        
        formatter_template = "Answer the question: {question}.  Dont make up answers or random text."

        prompt = PromptTemplate(
            input_variables=["question"], template=formatter_template
        )

        load_dotenv()
        llm = AzureMLOnlineEndpoint(
            endpoint_api_key=os.getenv("AZUREML_ENDPOINT_API_KEY"),
            endpoint_url=os.getenv("AZUREML_ENDPOINT_URL"),
            deployment_name=os.getenv("AZUREML_DEPLOYMENT_NAME"),
            model_kwargs={"temperature": 0.1, "max_tokens": 500},
            content_formatter=content_formatter
        )

        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run({"question": text})
        return response
    except requests.exceptions.RequestException as e:
        # Handle any requests-related errors (e.g., network issues, invalid URL)
        raise ValueError(f"Error with the API request: {e}")

    except json.JSONDecodeError as e:
        # Handle any JSON decoding errors (e.g., invalid JSON format)
        raise ValueError(f"Error decoding API response as JSON: {e}")
    except Exception as e:
        # Handle any other errors
        raise ValueError(f"Error: {e}")


def main():


    st.markdown('#')

    # Define a custom CSS class for the tooltip-like container
    st.markdown(
        """
        <style>
        .tooltip-container {
            position: relative;
            display: inline-block;
            cursor: pointer;
        }

        .tooltip-content {
            visibility: hidden;
            position: absolute;
            background-color: #f9f9f9;
            color: black;
            padding: 5px;
            border-radius: 3px;
            z-index: 1;
            top: -40px;
            left: 100%;
            width: 200px;
            text-align: left;
            white-space: normal;
        }

        .tooltip-container:hover .tooltip-content {
            visibility: visible;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Create a radio button for user selection
    selected_option = st.radio("Select an option:", ("langchain", "standard httprequest"))

    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

        # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if selected_option == "langchain":
                    response = askdocuments(text=prompt)
                    st.write(response)
                elif selected_option == "standard httprequest":
                    response = askdocuments2(question=prompt)

                    st.write(response)

        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)


if __name__ == "__main__":
    main()
