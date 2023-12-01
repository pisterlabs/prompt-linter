import openai
import streamlit as st
import os

# Set up API key

max_turns = 4
num_agents = 3

# Streamlit interface setup
st.title("Multi-Agent GPT App v2")
master_prompt = st.text_input("Enter Master Prompt:")
agent_prompts = [st.text_input(f"Enter Agent {i+1} Prompt:") for i in range(num_agents)]
final_prompts = [st.text_input(f"Enter Agent {i+1} Final Prompt:") for i in range(num_agents)]
# Define agent data structure
agents = [{'prompt': prompt, 'last_summary': '', 'messages': []} for prompt in agent_prompts]
i = 0
for agent in agents:
    agent['final_prompt'] = final_prompts[i]
    i += 1


def update_streamlit_interface(agents):
    """
    Updates the Streamlit interface with the current state of each agent and the group messages.

    Parameters:
    agents (list): A list of dictionaries, each representing an agent with keys 'prompt', 'last_summary', and 'messages'.
    """
    st.sidebar.title("Agent States")

    # Display each agent's prompt, last summary, and messages
    for i, agent in enumerate(agents):
        st.sidebar.subheader(f"Agent {i+1}")
        st.sidebar.text(f"Prompt: {agent['prompt']}")
        st.sidebar.text(f"Last Summary: {agent['last_summary']}")
        
        # You might want to limit the number of displayed messages, or implement some way of browsing through them if there are many
        st.sidebar.text("Messages:")
        for msg in agent['messages']:
            st.sidebar.text(msg)

    st.title("Group Interaction")
    
    # You might construct a single string or other data structure representing the group interaction from individual messages
    group_interaction = construct_group_interaction(agents)
    st.text_area("Group Interaction", group_interaction, height=300)  # Adjust height as needed

def construct_group_interaction(agents):
    """
    Constructs a string representing the group interaction from the messages of all agents.

    Parameters:
    agents (list): A list of dictionaries, each representing an agent with keys 'prompt', 'last_summary', and 'messages'.

    Returns:
    str: A string representing the group interaction.
    """
    messages = []
    if "says finally:" in agents[2]['messages'][-1]:
        messages = [agents[2]['messages'][-1]]
    elif "says finally:" in agents[1]['messages'][-1]:      
        messages = [agents[1]['messages'][-1]]
    elif "says finally:" in agents[0]['messages'][-1]:      
        messages = [agents[0]['messages'][-1]]

    else:
        messages = [agents[0]['messages'][-1]]
        #agent =  agents[0]
        #for msg in agent['messages']:
        #        messages.append(f"{msg}")
    
    return '\n'.join(messages)

# The agents variable would be something like:
# agents = [{'prompt': '...', 'last_summary': '...', 'messages': ['...']}, ...]

# You'd call update_streamlit_interface(agents) inside your main loop to update the interface at each turn
def parse_response(response):
    message_start = response.find("MESSAGE:")
    summary_start = response.find("SUMMARY:")

    if summary_start < message_start:
        message = response[message_start + 8:]
        summary = response[summary_start + 8:message_start]
    else:
        message = response[message_start + 8:summary_start]
        summary = response[summary_start + 8]

    return summary, message




if st.button("run"):

    # Main loop for turns
    for turn in range(max_turns):
        for i1, agent in enumerate(agents):
            # Construct prompt for GPT
            prompt = master_prompt + "\n" + agent['prompt'] + "\nhere's the summary you have made so far of the conversation\n" + agent['last_summary'] + "\nhere is the entire covnersation\n" + ''.join(agent['messages']) + "\n please summarize your understanding of the full conversaton first in bullet points (after you write \"SUMMARY:\"), then submit your message to the group after writing \"MESSAGE:\"\n"
            
            # Call GPT API
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                temperature=0.7,
                max_tokens= 400
            )
            
            # Parse response to get new summary and message
            summary, message = parse_response(response.choices[0].text.strip())
            
            # Update agent's last summary and messages
            agent['last_summary'] = summary
            for i2, a in enumerate(agents):
                a['messages'].append(f"Agent {i1+1} says: "+ message)
            
            # Update Streamlit interface
            update_streamlit_interface(agents)
            
            
    for i1, agent in enumerate(agents):
        # Construct prompt for GPT
        prompt = master_prompt + agent['prompt'] + agent['last_summary'] + ''.join(agent['messages']) + "\n" + agent['final_prompt']
            
            # Call GPT API
        response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                temperature=0.7,
                max_tokens= 400
            )
        message = response.choices[0].text.strip()
        agent['messages'] = [(f"Agent {i1+1} says finally: "+ message)]
            # Update agent's last summary and messages            
            # Update Streamlit interface
        update_streamlit_interface(agents)


