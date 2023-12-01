import time
import boto3
import streamlit as st
from langchain.chains import ConversationChain
from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory

#AWS S3 Configuration
s3_bucket_name = "your bucket name" #enter your S3 bucket name
s3_client = boto3.client("s3")

#Streamlit configuration

#set the background color of the webpage
st.write(
    f"<style>body {{background-color: #EOF2F1; }}</style>",
    unsafe_allow_html=True,
)

#attach an image to the webpage
st.image("images/smileyface.png", width=100)

#create a title 
st.write(
    f"<style>.title{{" +
    "border: 2px solid #008cBA; " + 
    "border-radius: 10px; " +
    "}}</stile>",
    unsafe_allow_html=True,
)

#st.markdown('<div class="title">ChatBedrock</div>',unsafe_allow_html=True)

st.title("ChatBedrock")

# Setup bedrock
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)

#Function to save conversation to S3 bucket 
def save_to_s3(message):
    timestamp = int(time.time())
    key=f"conversation/{timestamp}.txt"
    s3_client.put_object(
        Bucket=s3_bucket_name,
        Key=key,
        Body=message.encode('utf-8')
    )

@st.cache_resource
def load_llm():
    llm = Bedrock(client=bedrock_runtime, model_id="anthropic.claude-v2")
    llm.model_kwargs = {"temperature": 0.7, "max_tokens_to_sample": 2048}

    model = ConversationChain(llm=llm, verbose=True, memory=ConversationBufferMemory())

    return model


model = load_llm()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        st.write(f"User Prompt: {prompt}")

        # prompt = prompt_fixer(prompt)
        result = model.predict(input=prompt)

        st.write(f"Model Result: {result}")
        # Simulate stream of response with milliseconds delay
        for chunk in result.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    #Save the conversation to S3 
    conversation = f"User: {prompt}\nAssistant: {full_response}\n"
    save_to_s3(conversation)

    st.session_state.messages.append({"role": "assistant", "content": full_response})


