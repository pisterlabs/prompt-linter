import streamlit as st
import openai
from redlines import Redlines

#openai.api_key = open("API_key", "r").read()
openai.api_key = st.secrets["API_KEY"]


def send_click():
    if st.session_state.user != '':
        prompt = st.session_state.user

        # Define the bot role:
        bot_role = "You are AChatdemio, a bot that helps young researchers to write better research papers. You respond in a concise and academic style. Write short sentences, so that they are clear."

        # Define the ChatGPT model
        output = openai.ChatCompletion.create(
            # Define the model
            model="gpt-3.5-turbo",
            # Define the prompt - add the context
            messages = [
                {'role':'system', 'content':"{}".format(bot_role)},
                {'role':'user', 'content':"Rewrite the following: {}".format(prompt)}
            ],
            # Set the temperature to 0 to get the most stable results
            temperature = 0,
        )

        response = output['choices'][0].message.content

        st.write(response)


def academic():
    if st.session_state.user != '':
        prompt = st.session_state.user

        # Define the bot role:
        bot_role = "You are AChatdemio, a bot that helps young researchers to write better research papers. You respond in a clear, concise and academic style."

        # Define the ChatGPT model
        output = openai.ChatCompletion.create(
            # Define the model
            model="gpt-3.5-turbo",
            # Define the prompt - add the context
            messages = [
                {'role':'system', 'content':"{}".format(bot_role)},
                {'role':'user', 'content':"Rewrite the following text, delimited by triple backticks, in academic style: ```{}```".format(prompt)}
            ],
            # Set the temperature to 0 to get the most stable results
            temperature = 0,
        )

        response = output['choices'][0].message.content

        st.subheader("Text with corrections")
        diff = Redlines(prompt,response)
        st.markdown(diff.output_markdown, unsafe_allow_html=True)

        st.subheader("Rewritten text")
        st.write(response)

def bullet():
    if st.session_state.user != '':
        prompt = st.session_state.user

        # Define the bot role:
        bot_role = "You are AChatdemio, a bot that helps young researchers to write better research papers. You respond in a clear, concise and academic style."

        # Define the ChatGPT model
        output = openai.ChatCompletion.create(
            # Define the model
            model="gpt-3.5-turbo",
            # Define the prompt - add the context
            messages = [
                {'role':'system', 'content':"{}".format(bot_role)},
                {'role':'user', 'content':"Provide one paragraph of text from the following bullet point list, delimited by triple backticks: ```{}```".format(prompt)}
            ],
            # Set the temperature to 0 to get the most stable results
            temperature = 0,
        )

        response = output['choices'][0].message.content

        st.write(response)

def slides():
    if st.session_state.user != '':
        prompt = st.session_state.user

        # Define the bot role:
        bot_role = "You are AChatdemio, a bot that helps young researchers to write better research papers. You respond in a clear, concise and academic style."

        # Define the ChatGPT model
        output = openai.ChatCompletion.create(
            # Define the model
            model="gpt-3.5-turbo",
            # Define the prompt - add the context
            messages = [
                {'role':'system', 'content':"{}".format(bot_role)},
                {'role':'user', 'content':"Prepare text for powerpoint slides in form of bullet points from the following text, delimited by triple backticks. The bullet points should summarize the provided texts in a few sentences. The sentences should be short and summarization of the provided text. Avoid using full sentences when possible. End each sentence with a dot. The text: ```{}```".format(prompt)}
            ],
            # Set the temperature to 0 to get the most stable results
            temperature = 0,
        )

        response = output['choices'][0].message.content

        st.write(response)


def proofreading():
    if st.session_state.user != '':
        prompt = st.session_state.user

        # Define the bot role:
        bot_role = "You are AChatdemio, a bot that helps young researchers to write better research papers. You respond in a clear, concise and academic style."

        # Define the ChatGPT model
        output = openai.ChatCompletion.create(
            # Define the model
            model="gpt-3.5-turbo",
            # Define the prompt - add the context
            messages = [
                {'role':'system', 'content':"{}".format(bot_role)},
                {'role':'user', 'content':"Find errors in the following text, delimited by triple backticks: ```{}```. Only ouput the errors and suggest how they should be corrected.".format(prompt)}
            ],
            # Set the temperature to 0 to get the most stable results
            temperature = 0,
        )

        response = output['choices'][0].message.content

        st.write(response)

st.title(":blue[Achademio] ☕")
st.write("# Improve the style of your research papers with AI")


# Create another two Streamlit widgets st.text_input() and st.button() to facilitate the query interface to users. 
st.text_area("Provide your text:", key="user")
option = st.selectbox("Select what would you like the Achademio bot to do:", ('Rewrite in academic style', 'Write a paragraph from my bullet point list', 'Summarize text in bullet points for slides', 'Proofread my text and output errors'))

if 'Rewrite in academic style' in option:
    academic()
elif 'Write a paragraph from my bullet point list' in option:
    bullet()
elif 'Summarize text in bullet points for slides' in option:
    slides()
else:
    proofreading()