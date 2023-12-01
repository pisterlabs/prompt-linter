import streamlit as st

from helpers import use_custom_css, write_footer
from prompt_widget import exercise_area

st.set_page_config(
    page_title="Home — Prompt Engineering for Lawyers",
    layout="wide"
)

if 'api_success' not in st.session_state:
    st.session_state['api_success'] = False

use_custom_css()

st.title("Prompt Engineering for Lawyers")

st.header("What's this? :open_mouth:")

"""
It's a course to explore what prompt engineering is and its possibilities. 
At the end of the course, you will:
 
* Craft prompts that are clear, concise and effective in generating the response you want
* Discover new ways to make large language models work harder for you
* Evaluate the quality of the generated text
* Understand better how to effectively use large language models like ChatGPT, GPT4, Claude, 
Bard, PaLM and many others through prompt engineering.

This course is also focused on **application**. 
It is designed for lawyers and other professionals who want to use prompt engineering in a variety of tasks in the 
legal domain. 
The examples and the tasks are focused on the legal domain.

"""

st.header("Why should I take this course? :thinking_face:")

"""
If you are a lawyer, and you want to stay ahead of the curve in the ever-evolving world of legal technology,
this is a great way to learn about prompt engineering. (Impress your colleagues!)

If you are not a lawyer, but you want to know how prompt engineering and large language models can affect
legal work, this is a good way to gain exposure.

One of the most important features I wanted is the ability to *experiment*. The widgets in this course
allow a user to input any prompt, so you can follow the course, experiment with your own input or 
compare two prompts to compare how effective they are.

Let's have fun! :the_horns:
"""

st.header("What do I need? :nerd_face:")

"""
**The only requirement to do this course is an OpenAI key.**
The key is used to power the examples and exercises throughout the course.
There are several LLMs you can practice and demonstrate prompt engineering on, but OpenAI's ChatGPT
is the most popular one.

Just a quick heads up that you will incur charges on your OpenAI account when you use the examples and 
exercises in this course. However, don't worry too much, as it will take a lot of queries before you have to pay a 
significant amount of money. Each 1,000 tokens costs only $0.002.

As Theodore Roosevelt once said, "The only way to learn anything is to do it." 
So, don't be afraid to experiment and try new things. The more you use the examples and exercises, 
the better you will become at prompt engineering.

If you don't have an OpenAI Key, sign up [here](https://platform.openai.com/signup). Once you have verified
your account, you can generate an API key. You can find more 
[detailed instructions](https://www.howtogeek.com/885918/how-to-get-an-openai-api-key/) here. 

Then enter it below:
"""


def test_api_key():
    import openai
    from openai.error import AuthenticationError
    try:
        openai.api_key = st.session_state.openai_key
        openai.Model.list()
    except AuthenticationError:
        st.session_state["api_success"] = False
        return st.error(
            "An incorrect API Key was provided. You can find your API key at "
            "https://platform.openai.com/account/api-keys."
        )
    st.session_state["api_success"] = True
    return st.success("Success! You are good to go.", icon="🎉")


with st.form("openai_key_form"):
    st.subheader("Enter your OpenAI API Key")
    st.text_input("OpenAI API Key", placeholder="sk-...", key="openai_key")

    submitted = st.form_submit_button("Submit")

    if submitted:
        test_api_key()

    st.warning("""
You were redirected here because you haven't entered your API key. 
Please enter your API key in the form above and click "Submit" to continue browsing the website.
    """, icon="🚨")

"""
Once you have entered your API key, the form below should work.
"""

exercise_area("Test exercise area", long=False)

"""
**You do not need to be a lawyer to do this course.** The examples and exercises might concern legal knowledge and work,
but the fundamental concepts in prompt engineering are the same and can be applied elsewhere.
"""

st.header("How to use this course? :hammer:")

"""
This is a [Streamlit App](https://streamlit.io/), which is a fancy Python Web App.

Use the sidebar (the grey area) to navigate between chapters. If you don't see the sidebar, you may have to click
the arrow at the top left hand corner to expand it.

The course is best read from top to bottom, but you can also read the chapter which you are most interested. The content
is fairly self contained and do not require each other. 

"""

st.header("Who are you? :smiley_cat:")

"""
Check out:
* My [GitHub Profile](https://github.com/houfu)
* My [LinkedIn Profile](https://www.linkedin.com/in/hou-fu-ang-0a6851113/)
* My [Mastodon Profile](https://kopiti.am/@houfu)
* [My Blog, Love.Law.Robots.](https://www.lovelawrobots.com)

I have used and experimented with generative AI, ChatGPT and prompt engineering:
* [SG Law Cookies](https://cookies.your-amicus.app), a algorithmically generated daily legal news summariser
* [An experiment](https://houfu-chatgptlawcompare-main-jymt8s.streamlit.app/) to compare how 
data augmentation helps on various legal questions in a narrow area of law.
* I tried to [pump up my CV using ChatGPT](https://www.lovelawrobots.com/let-chatgpt-take-over-your-resume/).

Even so, I don't see myself as a data science expert or a prompt engineering expert.
Let's learn together!
"""

st.header("Feedback, suggestions etc")

"""
Your comments and feedback are much appreciated. 

Visit the GitHub repository for this website, and submit an issue.
"""

write_footer()
