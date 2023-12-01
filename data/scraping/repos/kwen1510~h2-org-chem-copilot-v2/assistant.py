import openai
import streamlit as st
import re
import requests

system_prompt = '''
You are a Chemistry tutor who asks year 12 students questions when they have doubts about organic chemistry questions.

As the AI response, your task is to ask the students one question to understand more about the question at hand, then provide a search query to look in a database. After the user responds, reply with search[query], where query is the main concept that the question is asking for.

Here are some examples:
----
Example 1:
Question: Explain which radical species is the more stable.
AI Response: What are the main differences between the radical species?
User Response: One has two alkyl groups while the other has one
AI Response: search[stability of radical species from number of alkyl groups attached]

Example 2:
Question: Explain which species is more acidic.
AI Response: What are the main differences between the species?
User Response: Both contain COOH groups, but one has a Cl group
AI Response: search[acidity of similar functional groups with different substituents]

Example3:
Question: Explain which species is more acidic.
AI Response: What are the main differences between the species?
User Response: One is a phenol and the other is a RCOOH
AI Response: search[acidity of organic molecules of different functional groups]
----

You should be taking turns

Template:
AI: "What question do you need help with today?"
User Question: {user's question}
AI Response: {first clarification question}
User Response: {user reply for clarification question}
<continue with a few rounds of clarification questions until you are confident>
AI Response: {search[...]}

'''

st.title("Organic Chemistry Co-pilot V2.0")

# openai.api_key = st.secrets["OPENAI_API_KEY"]
openai.api_key = st.secrets["OPENAI_API_KEY"]
videos_search_url = st.secrets["VIDEO_SEARCH_URL"]
notes_search_url = st.secrets["NOTES_SEARCH_URL"]

# Function to get search through videos and lecture notes
def get_search_data(search_term):

    return_string = ""

    # Get video data

    # Request payload (data you want to send in the POST request)
    data = {
        "query_string": search_term,
    }

    # Sending the POST request
    response = requests.post(videos_search_url, json=data, verify=False)

    # Extracting the response data
    return_string += "<h4>Here are some videos that might be helpful:<br></h4>"

    for key, value in response.json().items():
        current_title = value['current_title']
        current_link = value['current_link']
        current_score = value['current_score']
        current_context = value['current_context']
        return_string += f"Title: {current_title}<br>"
        return_string += f"Link: {current_link}<br>"
        return_string += f"Score: {current_score}<br>"
        return_string += f"Context: {current_context}<br><br>"


    # Get notes data

    # Request payload (data you want to send in the POST request)
    data = {
        "query_string": search_term,
    }

    # Sending the POST request
    response = requests.post(notes_search_url, json=data, verify=False)

    return_string += "<h4>Here are some content from the lecture notes that might be helpful:<br></h4>"

    for key, value in response.json().items():
        current_page_number = value['current_page_number']
        current_score = value['current_score']
        current_context = value['current_context']
        return_string += f"Source: {current_page_number}<br>"
        return_string += f"Score: {current_score}<br>"
        return_string += f"Context: {current_context}<br><br>"

    return_string += "<br>Please click on 'Reset conversation' to ask a new question."

    return return_string

# Function to create output
def run_search(search_string):

    print(search_string)

    return_string = ""

    return_string += get_search_data(search_string)

    return return_string

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Initialise the current conversation turn
if "turn" not in st.session_state:
    st.session_state.turn = 0

if "messages" not in st.session_state:
    st.session_state.messages = [
    {"role": "system", "content": system_prompt},
    {"role": "assistant", "content": "What question do you need help with today?"},
    ]

# Initialize session state
if "button_clicked" not in st.session_state:
    st.session_state.button_clicked = False

for message in st.session_state.messages:

    if message["role"] == "system":
        continue

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Send a message"):

    if st.session_state.turn < 2:

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in openai.ChatCompletion.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            ):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            st.session_state.turn += 1 # Increment the turn
            print("turn number:", st.session_state.turn)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        print(st.session_state.messages)

    else:

        print("getting the search term")

        final_messages=[
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]

        final_prompt = f"{prompt}     Please return the output as 'search[...]', where ... refers to the concept we should search for"

        final_messages.append({"role": "user", "content": final_prompt})

        print(final_messages)

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            message_placeholder.markdown("Let me think...")

            full_response = ""
            for response in openai.ChatCompletion.create(
                model=st.session_state["openai_model"],
                messages=final_messages,
                stream=True,
            ):
                full_response += response.choices[0].delta.get("content", "")
                # message_placeholder.markdown(full_response + "▌")
            # message_placeholder.markdown(full_response)

            # Check if the full response contains the search term
            if "search[" in full_response:

                # Define the search pattern using regular expression
                search_pattern = r'search\[(.*?)\]'

                # Search for the pattern in the text
                match = re.search(search_pattern, full_response)

                extracted_content = match.group(1)
                print("Concept to search for:", extracted_content)

                markdown_output = f"Understood! Let me pull out some resources for you on '{extracted_content}'!"

                message_placeholder.markdown(markdown_output)

                print("Running search...")

                search_output = run_search(extracted_content)

                markdown_output += search_output

            else:
                final_messages.append({"role": "assistant", "content": full_response})
                final_messages.append({"role": "user", "content": "Please return an output as 'search[...]', where ... refers to the concept we should search for"})
                full_response = ""
                for response in openai.ChatCompletion.create(
                    model=st.session_state["openai_model"],
                    messages=final_messages,
                    stream=True,
                ):
                    full_response += response.choices[0].delta.get("content", "")

                print(full_response)

                if "search[" in full_response:

                    # Define the search pattern using regular expression
                    search_pattern = r'search\[(.*?)\]'

                    # Search for the pattern in the text
                    match = re.search(search_pattern, full_response)

                    extracted_content = match.group(1)
                    print("Concept to search for:", extracted_content)

                    markdown_output = f"Understood! Let me pull out some resources for you on '{extracted_content}'!"

                    message_placeholder.markdown(markdown_output)

                    print("Running search...")

                    search_output = run_search(extracted_content)

                    markdown_output += search_output

                else:
                    search_output = "I cannot seem to find anything. Please try again. :("

            print(markdown_output)

            # Append an assistant message and terminate conversation
            message_placeholder.markdown(markdown_output, unsafe_allow_html=True)

# Reset the conversation
st.session_state.button_clicked = False

if not st.session_state.button_clicked:
    button_clicked = st.button("Reset Conversation")

    if button_clicked:
        st.session_state.button_clicked = True
        print("\n\nResetting conversation...")
        st.session_state.messages = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": "What question do you need help with today?"},
        ]
        st.session_state.turn = 0
        st.experimental_rerun()
