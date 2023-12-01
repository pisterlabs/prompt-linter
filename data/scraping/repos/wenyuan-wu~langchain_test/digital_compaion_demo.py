# import openai
# import streamlit as st
# from streamlit_pills import pills

#
# st.title("🦾 Assistant 👾")
# st.subheader("Usage: the `stream` *argument* 🌫️")
#
# # openai.api_key = st.secrets['api_secret']
# # API_O = st.text_input(
# #     ":blue[Enter Your OPENAI API-KEY :]",
# #     placeholder="Paste your OpenAI API key here (sk-...)",
# #     type="password",
# # )
#
# # You can also use radio buttons instead
# selected = pills("", ["NO Streaming", "Streaming"], ["🎈", "🌈"])
#
#
# # if API_O:
# #     openai.api_key = API_O
# user_input = st.text_input("You: ",placeholder = "Ask me anything ...", key="input")
#
# if st.button("Submit", type="primary"):
#     st.markdown("----")
#     res_box = st.empty()
#
#     if selected == "Streaming":
#         report = []
#         # Looping over the response
#         for resp in openai.Completion.create(model='text-davinci-003',
#                                             prompt=user_input,
#                                             max_tokens=120,
#                                             temperature = 0.5,
#                                             stream = True):
#             # join method to concatenate the elements of the list
#             # into a single string,
#             # then strip out any empty strings
#             report.append(resp.choices[0].text)
#             result = "".join(report).strip()
#             result = result.replace("\n", "")
#             res_box.markdown(f'*{result}*')
#
#     else:
#         completions = openai.Completion.create(model='text-davinci-003',
#                                             prompt=user_input,
#                                             max_tokens=120,
#                                             temperature = 0.5,
#                                             stream = False)
#         result = completions.choices[0].text
#
#         res_box.write(result)
# st.markdown("----")
#
# with st.sidebar:
#     # st.video('https://youtu.be/CqqELxWGUy8')
#     st.markdown(
#     '''
#     **Read my blog Post:**
#
#     [ *How to ‘stream’ output in ChatGPT style while using openAI Completion method*]()
#
#     *Codes are avaialble in this blog post.*
#     '''
#     )
# # Written at

"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message

from langchain.chains import ConversationChain
from langchain.llms import OpenAI


def load_chain():
    """Logic for loading the chain you want to use should go here."""
    llm = OpenAI(temperature=0)
    chain = ConversationChain(llm=llm)
    return chain

chain = load_chain()

from dotenv import load_dotenv

load_dotenv()

# From here down is all the StreamLit UI.
st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
st.header("LangChain Demo")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "Hello, how are you?", key="input")
    return input_text


user_input = get_text()

if user_input:
    output = chain.run(input=user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")

