import streamlit as st
import openai
from streamlit_extras.switch_page_button import switch_page

st.title("Onboarding")

OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']

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


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "This is a onboarding session of a goal tracking and journaling app. Spicy is sarcastic and roasty like GLaDOS, yet caring and encouraging as well. Daisy is gentle, inviting, and comforting. Spicy and Daisy's goal is to help the user set monthly, weekly goals and goal(s) for tomorrow, which will be tracked using the app. For this, they can ask the user about the details if needed. The reply should not be over 200 words in total."},
                                 {"role": "assistant", "content": 
"""**Daisy**:\n
"Hey there! Welcome to the Tearoom. Really glad you decided to join us. I'd suggest making yourself comfortable, maybe with a cup of tea, as we guide you through."

**Spicy**:\n
"Ready to get your life on track? Or at least pretend to? Daisy and I have been whipping slackers into shape for a while now. And I guess you need a nudge in the right direction, don't you?"

**Daisy**:\n
"Or perhaps, our friend here just wants a little space in their day to reflect and stay on track. If that's the case, you're in good company now. Before we start, let us know who you are…" """
}]
    
# Display chat messages from history on app rerun
for message in st.session_state.messages[1:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"].replace("[Instruction from now on: Focus on building specific goals and action plans with the user. Interact with them actively and try to understand them and their goals.]","").replace("[Instruction from now on: If the onboarding session is completed, add <EndOfSession> at the end of the response.]",""))

# React to user input
if "count" not in st.session_state: st.session_state.count = 0
if "inputHint" not in st.session_state: st.session_state.inputHint = "Enter your name here"

if prompt := st.chat_input(st.session_state.inputHint):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.count == 0:
        # Add user message to chat history
        username = prompt
        starting_conversation = {"role": "assistant", "content":f"""
**Spicy:**

"So, {username}, ready to tackle life head-on? Or just trying to remember what you had for breakfast?"

**Daisy**:

"Spicy! Be nice. This is a safe space for everyone, whether you're here to track daily thoughts, set goals, or just reflect."

**Spicy**:

"Right, right. I'm just making sure {username} is awake and with us."

**Daisy**:

"Let's start simple. How about jotting down a few things you'd like to achieve? Or perhaps any inconvenient feelings you've had lately that we could transform into a trackable goal? It's all about growth and understanding, so feel free to take your time."
 """}
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append(starting_conversation)
        st.session_state.count += 1
        st.session_state.inputHint = "Type here"
        st.experimental_rerun()
        
    elif st.session_state.count == 1:
        st.session_state.messages.append({"role": "user", "content": prompt})

    elif st.session_state.count == 2:
        st.session_state.messages.append({"role": "user", "content": prompt + "[Instruction from now on: Focus on building specific goals and action plans with the user. Interact with them actively and try to understand them and their goals.]"})
        # Add user message to chat history
    elif st.session_state.count == 3:
        st.session_state.messages.append({"role": "user", "content": prompt})
    else:
        st.session_state.messages.append({"role": "user", "content": prompt + "[Instruction from now on: If the onboarding session is completed, add <EndOfSession> at the end of the response.]"})
        
        
    # Get assistant response
    response = get_response(st.session_state.messages)

    # Check if onboarding session is completed
    completed = True if "<EndOfSession>" in response else False

    st.session_state.count += 1
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        if completed:              
            st.markdown(response.rstrip("<EndOfSession>"))
            if st.button("Continue"):
                switch_page("set_goal")
        else:
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
    # Add assistant response to chat history