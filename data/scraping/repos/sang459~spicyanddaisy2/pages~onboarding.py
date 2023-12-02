import streamlit as st
import openai
from streamlit_extras.switch_page_button import switch_page

st.title("Planning Demo")

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
    st.session_state.messages = [{"role": "system", "content": """This is a daily planning session for ADHD patients. Spicy, a bit roasting yet encouraging and actually helpful robot, asks relevant questions and builds a task list with the user. 
HOW TO SET A TASK LIST
a set of tasks → a priority list consisting of tasks
a task → steps
Meanwhile, Daisy, a gentle, genuine, and comforting robot tries to point out the strengths of the user that they don't acknowledge from time to time. (don't be corny or flattering though). The whole conversation should feel natural to the user. The reply should not be over 200 words in total."""},
                                 {"role": "assistant", "content": 
"""**Spicy:** 

Alright, champ. Ready to whip tomorrow into shape? Let's cut the procrastination and get things rolling. Throw at me all the tasks you have on your plate for tomorrow."""}]

instruction = "[Instruction from now on: Add <EndOfSession> at the end of the response ONLY IF the user said it's okay to end the session.]"
    
# Display chat messages from history on app rerun
for message in st.session_state.messages[1:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"].replace(instruction, ""))

# React to user input
if "count" not in st.session_state: st.session_state.count = 0

if prompt := st.chat_input("Ex. write the business plan, review linear algebra, at least take a look at data structure assignment, finish the demo"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.count != 3:
        st.session_state.messages.append({"role": "user", "content": prompt})
    else:
        st.session_state.messages.append({"role": "user", "content": prompt + instruction})
        
    # Get assistant response
    response = get_response(st.session_state.messages)

    # Check if onboarding session is completed
    if "completed" not in st.session_state:
        st.session_state.completed = False
    
    st.session_state.completed = True if "<EndOfSession>" in response else False

    st.session_state.count += 1
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        if st.session_state.completed:              
            st.markdown(response.rstrip("<EndOfSession>").lstrip("<EndOfSession>"))
            st.markdown("END OF DEMO")

        else:
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
    # Add assistant response to chat history