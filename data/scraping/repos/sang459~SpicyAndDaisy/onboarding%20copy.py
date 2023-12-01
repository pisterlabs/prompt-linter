import streamlit as st
import openai

OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
"""
def get_response(chat_history):
    eng_response = openai.ChatCompletion.create(
                    model= "gpt-4",
                    messages=chat_history,
                    stream=False,
                    temperature=0.91,
                    top_p = 0.93
                    )

    message = eng_response['choices'][0]['message']['content']

    return message
"""


st.title("Onboarding")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": 
"""**Daisy**:\n
"Hey there! Welcome to the Tearoom. Really glad you decided to join usl. I'd suggest making yourself comfortable, maybe with a cup of tea, as we guide you through."

**Spicy**:\n
"Ready to get your life on track? Or at least pretend to? Daisy and I have been whipping slackers into shape for a while now. And I guess you need a nudge in the right direction, don't you?"

**Daisy**:\n
"Or perhaps, our friend here just wants a little space in their day to reflect and stay on track. If that's the case, you're in good company now. Before we start, let us know who you are…" """
}]
    
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if "count" not in st.session_state: st.session_state.count = 0
if "inputHint" not in st.session_state: st.session_state.inputHint = "What's your name?"

if prompt := st.chat_input(st.session_state.inputHint):
    print(st.session_state.count)
    if st.session_state.count == 0:
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        username = prompt
        starting_conversation = {"role": "assistant", "content":f"""
**Spicy:**

"Ah, {username}, ready to tackle life head-on? Or just trying to remember what you had for breakfast?"

**Daisy**:

"Spicy! Be nice. This is a safe space for everyone, whether you're here to track daily thoughts, set goals, or just reflect."

**Spicy**:

"Right, right. I'm just making sure {username} is awake and with us."

**Daisy**:

"Let's start simple. How about jotting down a few things you'd like to achieve? Or perhaps any inconvenient feelings you've had lately that we could transform into a trackable goal? It's all about growth and understanding.
 """}
        st.session_state.messages.append({"role": "user", "content": prompt})
        #  + "\n[Spicy and Daisy's goal is to help the user set monthly, weekly goals and goal(s) for tomorrow. The reply should not be over 200 words in total.]"
        st.session_state.count += 1
        st.session_state.inputHint = "Type here"
        print('rerun 직전까지')
        st.experimental_rerun()
        
    elif st.session_state.count == 1:
        print("여기까지옴")
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Add user message to chat history
        # response = get_response(st.session_state.messages)
        response = 'k'
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

