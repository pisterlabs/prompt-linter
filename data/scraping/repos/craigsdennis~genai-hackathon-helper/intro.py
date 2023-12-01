from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI

from utils import get_function_body

# Load environment variables
load_dotenv()


@st.cache_resource
def get_client():
    return OpenAI()


# Streamlit can output HTML
st.title("👨‍🏫 Introduction to OpenAI (Part 1)")

# In a Streamlit app if you put an object on a line it will try to render it
# This is a string and it will show up as Markdown. (More in the Streamlit section below)
"""This is an interactive application using [Streamlit](https://streamlit.io) with the sole purpose
of getting you hacking on the [OpenAI APIs](https://platform.openai.com/) as quickly as possible.

##  🦙 LLMs - Large Language Models

Provides the ability to generate text based on context. Brought into the spotlight 
with ChatGPT in November of 2022.

- **G** enerative 🎨
- **P** re-trained 📚
- **T** ransformer 🧠

It does this using [tokens](https://platform.openai.com/tokenizer).

It's trained and gets better through [RLHF](https://en.wikipedia.org/wiki/Reinforcement_learning_from_human_feedback).

### OpenAI Models

OpenAI provides access to its [models](https://platform.openai.com/docs/models).

All you need is an [API Key](https://platform.openai.com/api-keys).

There is a [playground](https://platform.openai.com/playground) where you can explore.

### Other Models

It is worth noting that there are other LLMs that are available over an API.

Check out [Hugging Face](huggingface.co), [Replicate](https://replicate.com), and [Cloudflare](https://developers.cloudflare.com/workers-ai/get-started/rest-api/)

If you are interested in swapping models in and out of your applications, [LangChain](https://langchain.com) is a framework that abstracts things away very nicely.

But for this hackathon, let's stay focussed.

## 🔥 Streamlit

Okay before we get there though...let's chat a bit about [Streamlit](https://docs.streamlit.io/library/api-reference).

Streamlit takes a little bit to get used to, but I guarantee you will take advantage of how quickly you can build.

It also provides [caching](https://docs.streamlit.io/library/advanced-features/caching) and [statefulness](https://docs.streamlit.io/library/advanced-features/session-state).

Remember everything you see here is a Streamlit app!

## Chat Completions API

The [Chat Completions API](https://platform.openai.com/docs/guides/text-generation/chat-completions-api) 
takes an list of messages and returns a completion.

Explore it a bit by filling out this form 👇
"""
client = get_client()


def simple_user_input(client, user_input):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": user_input},
        ],
    )
    return response


def get_response_content(response):
    content = response.choices[0].message.content
    print(content)
    return content


with st.form("simple"):
    input = st.text_input("Prompt")
    submitted = st.form_submit_button("Send to OpenAI Chat Completion API")
    if submitted:
        st.code(get_function_body(simple_user_input, {"user_input": input}))
        with st.spinner("Sending completion request..."):
            response = simple_user_input(client, input)
            st.markdown(
                """
                #### The Response Object
                
                The raw API response looks lke this:
            """
            )
            st.json(response.model_dump_json())
            st.markdown(
                f"""
                👀 The `model` will always show the actual version: `"{response.model}"`
                
                The `response` object that is returned is the inflated response
                """
            )
            st.code(get_function_body(get_response_content))
            with st.expander("Output"):
                st.markdown(get_response_content(response))

"""
### Prompting

Prompting is an art form and is a new way of thinking. Let it amaze you. 

**PRACTICE** is important!

📚 [Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering/prompt-engineering)

### System messages

__IMHO__, the real magic of all these applications come down to System message.

You will use these to instruct the model how to behave.

You can not only control the personality, but you can also control what it should expect.

Let's show and not just tell 🪄
"""


def systemized_user_input(client, system_message, user_input):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_input},
        ],
    )
    return response


REGEX_SYSTEM_MESSAGE = """You are an instructor who is aware of how challenging regular expressions can be for learners.
You are patient and take your time to make sure important lessons are honed in.
A user will give you a regex and you will explain what it is doing, and how it might be used."""

with st.form("systemized"):
    system_message = st.text_area(
        label="System Message (would normally be hidden)",
        value=REGEX_SYSTEM_MESSAGE,
        height=7,
    )
    input = st.text_input("Regular Expression")
    submitted = st.form_submit_button("Send to OpenAI Chat Completion API")
    if submitted:
        st.code(
            get_function_body(
                systemized_user_input,
                {
                    "user_input": input,
                    "system_message": system_message,
                },
            )
        )
        with st.spinner("Sending completion request..."):
            st.balloons()
            response = systemized_user_input(client, system_message, input)
            st.markdown(get_response_content(response))


"""
### Embrace the API
 
Remember that the [messages parameter is a list](https://platform.openai.com/docs/guides/text-generation/chat-completions-api). If you keep track of what is returned you can add it as a `role` of `assistant`, you can have it refine.

You don't need to use this in a Chat interface, [be creative](https://platform.openai.com/examples)!

## But wait there's more!

Check out the [Official OpenAI Cookbook](https://cookbook.openai.com/)

You can actually have the LLM [call a function](https://platform.openai.com/docs/guides/function-calling) to get more information!

[LangChain](https://langchain.com) has a bit of a learning curve.

⏰Moar coming soon!
"""
