
# Importing required packages
import streamlit as st
import openai
import uuid
import json
import os
import time
from train_informations import get_journey,get_actual_time_and_date,get_best_prices
from openai import OpenAI
import logging



openai.api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI()
assistant_id = os.getenv('assistant_id')


# Initialize conversation history
conversation_history = []
# add you assistant_id 
st.title('DB GPT Train Assistant 🚉')
with st.expander("ℹ️ Disclaimer"):
    st.caption(
        "Please note, this demo is designed to process a maximum of 5 interactions."
    )

st.warning('Due to the beta stage of the Assistants API, processing messages can take a significant amount of time.', icon="⚠️")


with st.sidebar:
    st.title('DB GPT Train Assistant 🚉')
    st.divider()
    st.subheader('Example queries:')
    st.write('Give me a train connection for tomorrow 16:00 from Hamburg Hbf to Frankfurt ')
    st.write('Give me the cheapest connections for next week tuesday from Berlin to Hannover ')
    st.divider()
    st.write('📝 Showcase for OpenAI Assistant, read more under : [blogpost](https://medium.com/@sharmadave/openais-assistants-api-uncovered-building-a-real-time-train-assistant-integrated-with-multiple-eb8d69fb3bf4)')
    st.markdown('**Created by [David Sharma](http://david-sharma.de)**')

# map function names from the assistant to your Python functions
functions = {
    'get_journey':get_journey,
    'get_actual_time_and_date':get_actual_time_and_date,
    'get_best_prices': get_best_prices
 }

# Function to call the assistant required functions and return their outputs as JSON strings
def execute_required_functions(required_actions):
    try:
        tool_outputs = []
        for tool_call in required_actions.submit_tool_outputs.tool_calls:
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            
            # Call the corresponding Python function
            if func_name in functions:
                function = functions[func_name]
                result = function(**args)  

                # Serialize the function's output to JSON
                result_str = json.dumps(result)
                print(f'Result from {func_name} : {result}')
                # Add the result to the list of tool outputs
                tool_outputs.append({
                    "tool_call_id": tool_call.id,
                    "output": result_str,
                })
    except Exception as e:
        st.error("Sorry, I'm confused. Please refresh the Page (F5)")
    return tool_outputs



if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "run" not in st.session_state:
    st.session_state.run = {"status": None}
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retry_error" not in st.session_state:
    st.session_state.retry_error = 0

max_messages = (
    5  
)

if hasattr(st.session_state.messages,'data') and len(st.session_state.messages.data) >= max_messages:
    st.info(
        """Notice: The maximum message limit for this demo version has been reached.!
        Thank you for your understanding."""
    )
else: 
    if "assistant" not in st.session_state:


        # Load the previously created assistant
        st.session_state.assistant = openai.beta.assistants.retrieve(assistant_id)

        # Create a new thread for this session
        st.session_state.thread = client.beta.threads.create(
            metadata={
                'session_id': st.session_state.session_id,
            }
        )
        
    # If the run is completed, display the messages
    elif hasattr(st.session_state.run, 'status') and st.session_state.run.status == "completed":
        print(st.session_state.run.status)

        # Retrieve the list of messages
        st.session_state.messages = client.beta.threads.messages.list(
            thread_id=st.session_state.thread.id
        )

        for thread_message in st.session_state.messages.data:
            for message_content in thread_message.content:
                # Access the actual text content
                message_content = message_content.text
                annotations = message_content.annotations
        
        # Display messages
        for message in reversed(st.session_state.messages.data):
            if message.role in ["user", "assistant"]:
                with st.chat_message(message.role,avatar=f'{"👩‍🎨" if message.role=="user" else "🤖"}'):
                    for content_part in message.content:
                        message_text = content_part.text.value
                        st.markdown(message_text)

    if prompt := st.chat_input("How can I help you?"):
        with st.chat_message('user',avatar="👩‍🎨"):
            st.write(prompt)

        # Add message to the thread
        st.session_state.messages = client.beta.threads.messages.create(
            thread_id=st.session_state.thread.id,
            role="user",
            content=prompt
        )

        # Do a run to process the messages in the thread
        st.session_state.run = client.beta.threads.runs.create(
            thread_id=st.session_state.thread.id,
            assistant_id=st.session_state.assistant.id,
        )
        if st.session_state.retry_error < 3:
            st.rerun()
                        
    if hasattr(st.session_state.run, 'status'):
        
        print(st.session_state.run.status)

        if st.session_state.run.status == "requires_action":
            print(f'requried action', st.session_state.run.required_action)
            with st.chat_message('assistant',avatar="🤖"):
                st.write(f'Executing Action ...')

            # Get the tool outputs by executing the required functions
            tool_outputs = execute_required_functions(st.session_state.run.required_action)

            # Submit the tool outputs back to the Assistant
            st.session_state.run = client.beta.threads.runs.submit_tool_outputs(
                thread_id=st.session_state.thread.id,
                run_id=st.session_state.run.id,
                tool_outputs=tool_outputs
            )
            if st.session_state.retry_error < 3:
                st.rerun()


        # Handle the 'failed' status
        elif st.session_state.run.status == "failed":
            st.session_state.retry_error += 1
            with st.chat_message('assistant'):
                if st.session_state.retry_error < 3:
                    st.write("Run failed, retrying ......")
                    time.sleep(3)  # Longer delay before retrying
                    st.rerun()
                else:
                    st.error("FAILED: The OpenAI API is currently processing too many requests. Please try again later ......")

        # Handle any status that is not 'completed'
        elif st.session_state.run.status != "completed":
            with st.chat_message('assistant',avatar="🤖"):
                st.write(f'Thinking ...... ')
            # Attempt to retrieve the run again, possibly redundant if there's no other status but 'running' or 'failed'
            st.session_state.run = client.beta.threads.runs.retrieve(
                thread_id=st.session_state.thread.id,
                run_id=st.session_state.run.id,
            )
            if st.session_state.retry_error < 3:
                st.rerun()