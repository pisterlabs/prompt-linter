import streamlit as st
from dotenv import load_dotenv
import streamlit as st
import psycopg2
import os
import json
from langchain.chains import load_chain
load_dotenv()

st.title("miniAGI agent configurations :computer:")
st.info("This is a configuration page for the miniAGI agent. If the prompt templates or chains are useable in your selected module, they will be in the sidebar.")

# Database connection
def connect_db():
    return psycopg2.connect(
        dbname=os.getenv("PGVECTOR_DATABASE"),
        user=os.getenv("PGVECTOR_USER"),
        password=os.getenv("PGVECTOR_PASSWORD"), 
        host=os.getenv("PGVECTOR_HOST")
    )

# Save new template to database
def save_template(name, template):
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("INSERT INTO prompt_templates (name, template) VALUES (%s, %s) RETURNING id;", (name, template))
    id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return id

def fetch_templates():
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("SELECT id, name, template FROM prompt_templates;")
    templates = cur.fetchall()
    cur.close()
    conn.close()
    return templates

# Save new chain to database
def save_chain(name, config_data):
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("INSERT INTO chains (name, config_data) VALUES (%s, %s) RETURNING id;", (name, json.dumps(config_data)))
    id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return id

# Fetch existing chains from database
def fetch_chains():
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("SELECT id, name, config_data FROM chains;")
    chains = cur.fetchall()
    cur.close()
    conn.close()
    return chains





prompt_templates, chains, other = st.tabs(["Prompt Templates", "Chains", "Other"])

# Prompt Templates configuration tab
with prompt_templates:
    
    # Display existing templates
    existing_templates = fetch_templates()
    st.write("Existing Templates:")
        
    selected_existing_template = st.selectbox("View existing templates", existing_templates, format_func=lambda x: x[1])
        
    if selected_existing_template:
        with st.container():
            st.write("Template Content:")
            st.write(selected_existing_template[2])

    st.divider()

    new_name = st.text_input("Enter the name for your new template", "")
    new_template = st.text_area("Enter your new template", "")
    if st.button("Save Template"):
        save_template(new_name, new_template)
        st.success("Template saved successfully.")



# Chains configuration tab
with chains:
    # Display existing chains
    existing_chains = fetch_chains()
    st.write("Existing Chains:")
    
    selected_existing_chain = st.selectbox("View existing chains", existing_chains, format_func=lambda x: x[1])
    
    if selected_existing_chain:
        with st.container():
            st.write("Chain Configuration:")
            st.json(selected_existing_chain[2])  # Assuming config_data is in JSON

    st.divider()
    
    langchain_hub = load_chain("lc://chains/hello-world/chain.json")
    st.write("Available chains:")

    # Fields for new chain
    new_chain_name = st.text_input("Enter the name for your new chain", "")
    
    # This is a simplified example; you would have more detailed inputs based on your chain's configuration needs
    new_chain_prompt = st.text_input("Enter the prompt for your new chain", "")
    new_chain_llm = st.text_input("Enter the language model for your new chain", "")
    
    if st.button("Save Chain"):
        new_chain_config = {
            'prompt': new_chain_prompt,
            'llm': new_chain_llm
        }
        save_chain(new_chain_name, new_chain_config)
        st.success("Chain saved successfully.")




# Future Configurations implementation
with other:

    st.divider()
    st.write("Other configurations are not yet implemented.")
