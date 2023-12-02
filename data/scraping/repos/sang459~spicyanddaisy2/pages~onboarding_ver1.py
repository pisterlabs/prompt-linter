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

instruction = "[Instruction from now on: Ask the user relevant questions, and work with them step by step to come up with a chart of actionable and specific daily goals for this week which will be tracked with this app, add <EndOfSession> at the end of the response.]"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "This is a onboarding session of a goal-tracker plus self-care app for burnout syndrome. Spicy is roasty yet caring and encouraging. Daisy is gentle, inviting, and comforting. Step by step, focus on understanding the user's problem. First, relevant questions about the symptoms and their duration. Second, ask questions about the causes of stress and burnout (take at least three questions and answers to dive deep into it). Third, questions about the user's lifestyle(e.g., sleep difficulties, exercise, alcohol consumption) and any negative thoughts. Finally, create a chart of daily goals for this week, which will be tracked with this app. Keep the reply under 200 words. Also, keep Daisy and Spicy in their characters and responsive to each other."},
                                 {"role": "assistant", "content": 
"""**Daisy:** 

Welcome to Tearoom! 🌼 It's a true pleasure to meet you. I see a journey ahead of us, where we'll uncover the strength and resilience you might not even know you possess.

**Spicy:**

Hey! 🌶️ Diving into the self-care pool, are we? Good move. Daisy's all about the pep talks, while I'm here for the straightforward, no-nonsense advice.

**Daisy:** 

Before we delve deeper, can we have your name?"""}]
    

# Display chat messages from history on app rerun
for message in st.session_state.messages[1:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"].replace(instruction,""))

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
        starting_conversation = {"role": "assistant", "content":f"""***Daisy:***
                                 
Welcome to your first teatime, {username}! ☕ Before we jump into the heart of things, let's take a moment to reflect on where you currently stand. Every journey begins with understanding the starting point.

***Spicy:***

Right. We're not about vague aspirations here. We're diving deep. So, spill the tea. What's been on your mind? What's been challenging you lately?"""}
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append(starting_conversation)
        st.session_state.count += 1
        st.session_state.inputHint = "Type here"
        st.experimental_rerun()
    elif st.session_state.count == 5:
        st.session_state.messages.append({"role": "user", "content": prompt + instruction})
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})

        
        
    # Get assistant response
    response = get_response(st.session_state.messages)

    # Check if onboarding session is completed
    completed = True if "<EndOfSession>" in response else False

    st.session_state.count += 1
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        if completed:              
            st.markdown(response.rstrip("<EndOfSession>").lstrip("<EndOfSession>"))
            if st.button("Continue"):
                switch_page("set_goal")
        else:
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
    # Add assistant response to chat history
