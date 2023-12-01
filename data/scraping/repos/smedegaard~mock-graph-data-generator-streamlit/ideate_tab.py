import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
from logic.agraph_conversions import convert_agraph_to_arrows, agraph_data_from_response
import logging
import openai
import json
import io

CHATGPT_KEY = "chatgpt"
LAST_PROMPT_KEY = "last_prompt"
LAST_OPENAI_RESPONSE = "last_openai_response"

def agraph_data_prompt(prompt: str)-> str:
    # Full prompt string to query openai with and finasse expected response
    full_prompt = f"""
    Given a prompt, extrapolate as many relationships as possible from it and provide a list of updates.

    If an update is a relationship, provide [ENTITY 1, RELATIONSHIP, ENTITY 2]. The relationship is directed, so the order matters.

    Each entity must be singular. Change any plural nouns to singular.

    Each relationship must have 3 items in the list.
    Limit the number of relationships to 12.

    Example:
    prompt: Alice is Bob's roommate. Bob is friends with Charlie.
    updates:
    [["Alice", "ROOMMATE", "Bob"], ["Bob", "FRIEND_OF", "Charlie"]]

    prompt: People who are friends with Alice are also friends with Bob.
    updates:
    [["Person", "FRIEND_OF", "Alice"], ["Person", "FRIEND_OF", "Bob"]]

    prompt: {prompt}

    updates:
    """
    return full_prompt

def generate_openai_response(prompt)-> str:
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.5,
        max_tokens=800,
        top_p=1.0,
        frequency_penalty=0.8,
        presence_penalty=0.0
        )
    # TODO: Validate reponse
    print(f'OpenAI Response: {response}')
    return response.choices[0].text
    
def ideate_tab():
    with st.expander("Instructions"):
        st.markdown(
            """
        Not sure how to start with data modeling? Use this variation of GraphGPT to generate a graph data model from a prompt.
        1. Add your [OpenAI API Key](https://platform.openai.com/account/api-keys) to the OpenAI API Key field
        2. Enter in a prompt / narrative description of what you'd like modelled
        3. Download data as an arrows.app compatible JSON file
        4. Proceed to the '② Design' tab
        """
        )

    # Configure ChatGPT
    if CHATGPT_KEY not in st.session_state:
        # For dev only
        # st.session_state[CHATGPT_KEY] = st.secrets["OPENAI_API_KEY"]
        st.session_state[CHATGPT_KEY] = ""
    if LAST_PROMPT_KEY not in st.session_state:
        st.session_state[LAST_PROMPT_KEY] = ""
    if LAST_OPENAI_RESPONSE not in st.session_state:
        st.session_state[LAST_OPENAI_RESPONSE] = ""

    current_api = st.session_state[CHATGPT_KEY]
    new_api = st.text_input("OpenAI API Key", value=current_api, type="password")
    if new_api != current_api:
        st.session_state[CHATGPT_KEY] = new_api

    openai.api_key = st.session_state[CHATGPT_KEY]

    if openai.api_key is None or openai.api_key == "":
        st.warning("Please enter your OpenAI API Key")
    else:
        # Display graph data from prompt
        prompt = st.text_input("Prompt")
        last_prompt = st.session_state[LAST_PROMPT_KEY]

        # Display Graph
        if prompt is not None and prompt != "":
            
            if prompt != last_prompt:
                print(f'New Prompt: {prompt}')
                st.session_state[LAST_PROMPT_KEY] = prompt
                full_prompt = agraph_data_prompt(prompt)
                openai_response = generate_openai_response(full_prompt)
                print(f'new open_ai_response: {openai_response}')
                st.session_state[LAST_OPENAI_RESPONSE] = openai_response

            oai_r = st.session_state[LAST_OPENAI_RESPONSE]
            # Convert openai response to agraph compatible data
            nodes, edges, config = agraph_data_from_response(oai_r)



            # Button to download graph data in arrows.app compatible JSON
            if nodes is not None:

                # Arrows compatible file
                arrows_dict = convert_agraph_to_arrows(nodes, edges)

                # Convert dict to file for download
                json_data = json.dumps(arrows_dict)
                json_file = io.BytesIO(json_data.encode())

                c1, c2 = st.columns([4,1])
                with c1:
                        agraph(nodes=nodes, 
                            edges=edges, 
                            config=config)
                with c2:
                    st.write("Download Options")
                    st.download_button("Arrows.app Compatible File", json_file, file_name="graph_data.json", mime="application/json")