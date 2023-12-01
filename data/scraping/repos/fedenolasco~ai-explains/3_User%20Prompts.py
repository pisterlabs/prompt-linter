import streamlit as st
import time
import os
import json
from dotenv import load_dotenv
import pandas as pd
import openai
import tiktoken
import time
from datetime import datetime
from PIL import Image
from elasticsearch import Elasticsearch, exceptions as es_exceptions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv() # Load environment variables from .env file

# get the current index name from the environment variable
index_name = os.getenv("ES_INDEX")

# Streamlit resources
# https://docs.streamlit.io/library/api-reference/layout
# https://github.com/blackary/st_pages
# https://towardsdatascience.com/5-ways-to-customise-your-streamlit-ui-e914e458a17c

favicon = Image.open("images/robot-icon2.png")
#st.set_page_config(page_title='AI-augments', page_icon=favicon, layout="wide")

# Read the contents from the CSS file
with open("css/styles.css", "r") as f:
    css = f.read()

# Include the CSS in the Streamlit app
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

col1, col2= st.columns(2)

def connect_to_elasticsearch():
    # Retrieve environment variables
    es_ca_cert_path = os.getenv("ES_CA_CERT_PATH")
    es_user = os.getenv("ES_USER")
    es_password = os.getenv("ES_PASSWORD")
    
    # Connect to the Elasticsearch cluster
    es = Elasticsearch("https://localhost:9200", 
                       ca_certs=es_ca_cert_path,
                       basic_auth=(es_user, es_password))
    
    try:
        # Try to get info from the Elasticsearch cluster
        info = es.info()
        print("Successfully connected to Elasticsearch!")
        print("Cluster Info:", info)
        return es, True
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, False


def disconnect_from_elasticsearch(es):
    if es is not None:
        try:
            es.transport.connection_pool.close()
            print("Disconnected from Elasticsearch.")
            return True
        except Exception as e:
            print(f"An error occurred while disconnecting: {e}")
            return False
    else:
        print("No active Elasticsearch client to disconnect.")
        return False

# Function to calculate similarity score
def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity[0][0]


# Get Windows username
username = os.getlogin()
st.session_state.username = username

load_dotenv()  # Load environment variables from .env file
openai.api_key = os.getenv("OPENAI_API_KEY")

# load helper file for prompt category
dfpromptcategory = pd.read_csv('./helpers/promptcategory.csv')

# Extract the 'subtaskid' column and convert it to a list
promptcategory_list = dfpromptcategory['subtaskid'].tolist()


# Check and initialize cumulative_cost in session state
if 'cumulative_cost' not in st.session_state:
    st.session_state.cumulative_cost = 0

# Get all the JSON files from the "collections" subfolder
collection_files = [f for f in os.listdir("collections") if f.endswith('.json')]

# Sort files by last modification time
collection_files.sort(key=lambda x: os.path.getmtime(os.path.join("collections", x)), reverse = True)

# Load the collections into a dictionary
collections = {}
for file in collection_files:
    with open(os.path.join("collections", file), 'r') as f:
        collection = json.load(f)
        collections[collection['collectionid']] = collection

# Create a dataframe to store the model information
data = {
    'model': ['gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-4', 'gpt-4-32k'],
    'max_tokens': [4096, 16384, 8192, 32768],
    'ui_select': [True, True, False, False],
    'tokens_input_per_1K': ['$0.0015', '$0.0030', '$0.0300', '$0.06'],
    'tokens_output_per_1K': ['$0.0020', '$0.0040', '$0.0600', '$0.12'],
    'description': [
        'Most capable GPT-3.5 model and optimized for chat at 1/10th the cost of text-davinci-003. Will be updated with latest model iteration 2 weeks after it is released. This project was developed using model release of 13 June 2023.',
        'Same capabilities as the standard gpt-3.5-turbo model but with 4 times the context. This project was developed using model release of 13 June 2023.',
        'More capable than any GPT-3.5 model, able to do more complex tasks, and optimized for chat. Will be updated with latest model iteration 2 weeks after it is released. This project was developed using model release of 13 June 2023.',
        'Same capabilities as the base gpt-4 mode but with 4x the context length. Will be updated with latest model iteration. This project was developed using model release of 13 June 2023.'
    ],
    'parameters': ['175B', '175B', '1.75T', '1.75T'],
}

df = pd.DataFrame(data)

# Page calculation variables and assumptions see readme.md for details
estimate_words_per_page = 500

# This function is from the OpenAI cookbook and is used to count the number of tokens in a message
# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning Token Estimate: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning Token Estimate: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

# This function is from the OpenAI cookbook and is used to send a message to the OpenAI API
def send_to_openai(tab_num, user_prompt, selected_model="gpt-3.5-turbo"):
    if tab_num >= len(st.session_state.system_prompts):
        raise ValueError("Invalid tab number.")
    system_message = {"role": "system", "content": st.session_state.system_prompts[tab_num]}
    user_message = {"role": "user", "content": user_prompt}
    messages = [system_message, user_message]

    # Display a spinner while the API call is processing
    with st.spinner('Processing...'):
        response = openai.ChatCompletion.create(
            model=selected_model,
            messages=messages
        )

    return response.choices[0].message['content'], response.usage['prompt_tokens'], response.usage['completion_tokens']

def send_to_openai2(tab_num, user_prompt, selected_model="gpt-3.5-turbo"):
    if tab_num >= len(st.session_state.system_prompts):
        raise ValueError("Invalid tab number.")
        
    system_message = {"role": "system", "content": st.session_state.system_prompts[tab_num]}
    user_message = {"role": "user", "content": user_prompt}

    # Display a spinner while the API call is processing
    with st.spinner('Processing...'):
        response = openai.ChatCompletion.create(
            model=selected_model,
            messages=[system_message, user_message]
        )

    model_response = response.choices[0].message['content']
    prompt_tokens = response.usage['prompt_tokens']
    completion_tokens = response.usage['completion_tokens']
    
    # Create a dictionary to hold the original prompt and the model's response
    conversation_piece = {
        "original_prompt": {
            "system": system_message["content"],
            "user": user_message["content"]
        },
        "model_response": model_response
    }
    
    return model_response, conversation_piece, prompt_tokens, completion_tokens



# This is an internal function that calculates the cost of the API call for input tokens
def calculate_cost(tokens_used, selected_model, data):
    # Get the index of the selected model
    model_index = data['model'].index(selected_model)
    
    # Retrieve the cost per 1K tokens for the model
    cost_per_1K = data['tokens_input_per_1K'][model_index]
    
    # Convert the cost to a float
    cost_per_1K_float = float(cost_per_1K.replace("$", ""))
    
    # Calculate the estimated cost
    estimated_cost = tokens_used * cost_per_1K_float / 1000
    
    return estimated_cost


# Model selection in the sidebar
st.sidebar.markdown("### Model Selection", help="List of LLM API models available, which differ in costs, context window, speed and reasoning capabilities. These LLM models can be extended to powerful Open LLM models that perform great 🚀 on particular tasks. Please refer also to [Huggingface Models](https://huggingface.co/models).")
selected_model = st.sidebar.selectbox("Select a model:", df['model'].tolist())

# Display the description for the selected model
st.sidebar.markdown(f"{df[df['model'] == selected_model]['description'].iloc[0]}")

# Display the max tokens and cost information for the selected model
st.sidebar.markdown(f"**Parameters:** {df[df['model'] == selected_model]['parameters'].iloc[0]} | **Max Tokens:** {df[df['model'] == selected_model]['max_tokens'].iloc[0]}")
#st.sidebar.markdown(f"**Max Tokens:** {df[df['model'] == selected_model]['max_tokens'].iloc[0]}")
st.sidebar.markdown(f"**Input Token Cost per 1K:** {df[df['model'] == selected_model]['tokens_input_per_1K'].iloc[0]}")
st.sidebar.markdown(f"**Output Token Cost per 1K:** {df[df['model'] == selected_model]['tokens_output_per_1K'].iloc[0]}")
st.sidebar.divider()
st.sidebar.markdown("### Prompt Selection", help="🚀 Boost your project efficiency with ready-to-use LLM prompt templates, specifically designed to elevate your productivity. Whether you're into data engineering, crafting technical documentation, or mastering business communication, these customizable templates have got you covered.")


# Step 1: Create a new list to store sorted collections
sorted_collections = []

# Step 2: Iterate over sorted collection_files
for file in collection_files:
    with open(os.path.join("collections", file), 'r') as f:
        collection = json.load(f)
        sorted_collections.append(collection)


# Step 3: Create a dictionary to map the concatenated collection name and ID to the original collection_id
collection_name_id_mapping = {f"{col['collectionname']} ({col['collectionid']})": col['collectionid'] for col in sorted_collections}

# Create a list of concatenated collection names and IDs for the select box
concatenated_collection_names_ids = list(collection_name_id_mapping.keys())

# Update the select box to use the sorted concatenated collection names and IDs
selected_collection_concatenated = st.sidebar.selectbox("Select a collection", concatenated_collection_names_ids, help="ℹ️ Select An Existing Collection allows you to quickly access a curated set of LLM prompt templates. You can choose from existing collections tailored for specific tasks, or even create your own. To keep things simple and effective, each collection contains a maximum of 5 prompt templates. This ensures that you can focus on a handful of highly relevant tasks.")

# Retrieve the original collection_id based on the selected concatenated collection name and ID
selected_collection_id = collection_name_id_mapping[selected_collection_concatenated]

selected_collection = collections[selected_collection_id]

tabs = [message["title"] for message in selected_collection["usermessages"]]

# Check if the selected collection is different from the last known selection
# Why do we need this? Because we want to reset the active tab to the first tab of the new collection
if "last_selected_collection" not in st.session_state or st.session_state.last_selected_collection != selected_collection_id:
    # Update the last known selected collection
    st.session_state.last_selected_collection = selected_collection_id
    
    # Reset the execute_button_pressed flag
    # st.session_state.execute_button_pressed = False
    
    # Reset the active tab to the first tab of the new collection
    st.session_state.current_tab = tabs[0]

# Initialize system_prompts in st.session_state if it doesn't exist
if "system_prompts" not in st.session_state:
    st.session_state.system_prompts = []

# Set the default system prompt
default_system_prompt = selected_collection["systemmessage"]

# Set the default system prompt
default_userskillfocus = selected_collection["userskillfocus"]

# Map the tabs to the user prompts
tabs = [message["title"] for message in selected_collection["usermessages"]]

# Set session_state.system_prompts to the system prompts from the selected collection
st.session_state.system_prompts = [selected_collection["systemmessage"] for _ in selected_collection["usermessages"]]

# Set the default directive prompt
default_directive_prompts = [message["directive"] for message in selected_collection["usermessages"]]

# Set the default task prompt
default_task_prompts = [message["task"] for message in selected_collection["usermessages"]]

# Set the default usage
default_usage = [message["usage"] for message in selected_collection["usermessages"]]

# Set the default promptcategory
default_promptcategory = [message["promptcategory"] for message in selected_collection["usermessages"]]

# Set the default usermessage_id
default_message_id = [message["id"] for message in selected_collection["usermessages"]]


# Initialize the current tab if not already initialized
if "current_tab" not in st.session_state:
    st.session_state.current_st.session_state.current_tab = tabs[0]

if "current_tab" not in st.session_state:
    st.session_state.current_tab = tabs[0]

# Navigation buttons for each tab in the sidebar
for t in tabs:
    if st.sidebar.button(t, key=f"nav_button_{t}"):
        st.session_state.current_tab = t

# Check if the current tab is different from the last known selection
if "last_selected_tab" not in st.session_state or st.session_state.last_selected_tab != st.session_state.current_tab:
    # Update the last known selected tab
    st.session_state.last_selected_tab = st.session_state.current_tab
    # Reset the execute_button_pressed flag
    # st.session_state.execute_button_pressed = False

# Initialize the current tab
tab = st.session_state.current_tab


with col1: 
    # Display the collection name
    st.markdown(f"# {selected_collection['collectionname']}")

    collection_model = selected_collection['collectionmodel']
    output_string = ", ".join(collection_model)

    st.markdown(f"{selected_collection['collectionusage']} This collection has been tested using [{output_string}] model(s). You are currently running the following template from this collection:")

    st.markdown(f"<h2 style='color: orange;'>{st.session_state.current_tab}</h2>", unsafe_allow_html=True)

    #  Set formatting for the system prompt
    formatted_system_prompt = [
        {"role": "system", "content": default_system_prompt}
    ]

    # Determine the number of tokens in the system prompt
    tokens_system = num_tokens_from_messages(formatted_system_prompt, selected_model)
    st.markdown("### System Prompt", help="(How the model should behave)")
    st.markdown(f":orange[{default_system_prompt}]")
    st.caption(f"(About {tokens_system} system prompt tokens counted using __tiktoken__ library from OpenAI.)")

# Check if 'current_tab' is in session_state before accessing it
# if "current_tab" not in st.session_state:
#     st.session_state.current_tab = tabs[0]  # Replace with the default tab you'd like to display

with col1: 
    # Display the user prompt header
    st.markdown("### User Prompt", help = "This is the user prompt, which is in this UI a combination of Directive + Task inputs.")

    with st.expander("Usage", expanded=False):
        # Display the usage right below the User Prompt
        st.markdown(f"__Usage:__ {default_usage[tabs.index(tab)]}", help="This is metadata information about the usage of this prompt template.")
        
    # Display the directive and task prompts
    directive = default_directive_prompts[tabs.index(tab)]
    task = default_task_prompts[tabs.index(tab)]
    promptcategory = default_promptcategory[tabs.index(tab)]
    message_id=default_message_id[tabs.index(tab)]
    st.markdown(f"__Prompt Category__: {promptcategory}", help="Pre-trained models can execute many tasks. This label identifies the actual LLM task.")
    
    if len(directive) > 1000:
        with st.expander("Directive", expanded=False):
            st.markdown(f":orange[__Directive__:] {directive}", help="(Main instructions such as format, expected output)")
    else:
        st.markdown(f":orange[__Directive__:] {directive}", help="(Main instructions such as format, expected output)")
    
    st.markdown(f":orange[__Task__: (Actual Data Input and Context information)]", help="(Actual Data Input and Context information)")
    user_task = st.text_area(label="Task:", value=task, key=f"user_task_{tab}", label_visibility='collapsed', height=50)

    # Combine the directive and task into a single user prompt
    user_prompt = directive + " " + user_task

    # Retrieve the max token limit for the selected model from the DataFrame
    max_tokens_row = df[df['model'] == selected_model]
    if not max_tokens_row.empty:
        max_tokens = int(max_tokens_row['max_tokens'].iloc[0])
    else:
        max_tokens = None

    # OpenAI example token count from the function defined above
    # https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
    # This section merges the system prompt and user prompt into a single list of messages
    formatted_user_prompt = [
        {"role": "system", "content": default_system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    with st.expander("Input Token Information", expanded=False):
        # This section counts the number of tokens in the system prompt and user prompt
        tokens_used = num_tokens_from_messages(formatted_user_prompt, selected_model)
        tokens_user = tokens_used - tokens_system
        costs = calculate_cost(tokens_used, selected_model, data)
        st.markdown(f"{tokens_used} input tokens counted (incl. #{tokens_system} tokens for system prompt and #{tokens_user} tokens for user prompt) using __tiktoken__ library from [OpenAI Platform](https://platform.openai.com/tokenizer). The cost of these input tokens is estimated for the chosen __{selected_model}__ model to be ${costs:.6f} (USD).")


        # Display the message explaining the token limits
        max_words = round(max_tokens * 0.75)
        estimated_pages = round(max_words / estimate_words_per_page)
        used_words = round(tokens_used * 0.75)
        remaining_words = round(max_words - used_words)
        token_url="[tokens](https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them)"
        if max_tokens:
            remaining_tokens = max_tokens - tokens_used
            can_execute = True
            btn_label = "Execute API Request"
            if tokens_used >= max_tokens:
                can_execute = False
                btn_label = "Cannot Execute API Request"
                # Display warning in red if tokens_used exceeds or matches the max_tokens
                st.markdown(f"<span style='color:red'>Warning: Your input prompt uses {tokens_used} {token_url} which meets or exceeds the limit of {max_tokens} tokens (~{estimated_pages} [pages](https://learnpar.com/word-count-calculator/)) for the selected model. Please reduce the size of input prompt.</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"Depending on the model used, API requests can use up to {max_tokens} {token_url} (~{max_words} words or ~{estimated_pages} [pages](https://learnpar.com/word-count-calculator/)) shared between input prompt and completion message. If your input prompt is {tokens_used} token (~{used_words} words), your completion can be {remaining_tokens} tokens (~{remaining_words} words) at most.")
        else:
            st.markdown("The selected model's {token_url} limit is unknown. Please ensure you stay within the token limit for the model you're using.")

    c1, c2 = st.columns(2)
    
    with c1:
        # Toggle button to enable or disable Elasticsearch memory usage
        es = None
        if os.getenv("ES_INSTALLED") == "True":
            # Connect to Elasticsearch  
            es, es_connected = connect_to_elasticsearch()
        else:
            # Disconnect from Elasticsearch
            es_connected = disconnect_from_elasticsearch(es)

        # Now, you can use `es` for queries if `es_connected` is True
        if es_connected:
            st.write("Connected to Elasticsearch.")
        else:
            st.write("Not Connected to Elasticsearch.")

    # Prepare Processing section
    st.sidebar.divider()

    # Initialize session state variables
    if 'execute_button_pressed' not in st.session_state:
        st.session_state.execute_button_pressed = False
    if 'result' not in st.session_state:
        st.session_state.result = None
    if 'conversation_piece' not in st.session_state:
        st.session_state.conversation_piece = None
    # Initialize session state for edited_result if it doesn't exist
    if 'edited_result' not in st.session_state:
        st.session_state.edited_result = None

    # Create a tab selector with different tab names
    special_tabs = ["Show Response", "Edit Response"]

    # Initialize session state variables
    if 'my_special_tab' not in st.session_state:
        st.session_state.my_special_tab = "Show Response"  # set the default value


    # This section executes the API call and return the result, prompt tokens and completion tokens
    with c2:
        if st.button(btn_label, key=f"execute_button_{tab}") and can_execute:
            
            st.session_state.execute_button_pressed = True
            # st.session_state.edited_result = None
            
            start_time = time.time()                
            result, conversation_piece, prompt_tokens, completion_tokens = send_to_openai2(tabs.index(tab), user_prompt, selected_model)
            
            st.session_state.result = result  # Assuming 'result' is updated here
            st.session_state.edited_result = result # When making a new call set edited response
            st.session_state.conversation_piece = conversation_piece # Assuming 'conversation_piece' is updated here
            
            elapsed_time = time.time() - start_time
            minutes, seconds = divmod(elapsed_time, 60)
            st.sidebar.write(f"Processing took {seconds:.2f} seconds.")
            st.sidebar.markdown(f"Prompt tokens used: {prompt_tokens}")
            st.sidebar.markdown(f"Completion tokens delivered: {completion_tokens}")
            # Extract pricing data for the selected model from df
            input_token_cost = df[df['model'] == selected_model]['tokens_input_per_1K'].iloc[0]
            output_token_cost = df[df['model'] == selected_model]['tokens_output_per_1K'].iloc[0]

            # Convert the cost to float (assuming the costs in df are given in the format "$x.xx")
            input_token_cost_float = float(input_token_cost.replace("$", ""))
            output_token_cost_float = float(output_token_cost.replace("$", ""))

            # Calculate the cost for prompt_tokens and completion_tokens
            prompt_cost = prompt_tokens * input_token_cost_float / 1000
            completion_cost = completion_tokens * output_token_cost_float / 1000
            completion_cost = round(completion_cost,6)
            prompt_cost = round(prompt_cost,6)

            # Sum the costs to get the total cost
            total_cost = prompt_cost + completion_cost
            total_cost = round(total_cost,6)
            # Update the global cumulative cost
            st.session_state.cumulative_cost += total_cost


            # Display the total cost in the Streamlit sidebar
            st.sidebar.markdown(f"___Total cost of this API call: ${total_cost:.6f} USD___")
            # Display the cumulative cost on the sidebar
            st.sidebar.markdown(f"___Cumulative cost of all API calls for this session: ${st.session_state.cumulative_cost:.6f} USD___")
            
            st.session_state.prompt_tokens = prompt_tokens
            st.session_state.completion_tokens = completion_tokens
            st.session_state.prompt_cost = prompt_cost
            st.session_state.completion_cost = completion_cost
            st.session_state.total_cost = total_cost

    if st.session_state.execute_button_pressed:
        with col2:
                with st.container():
                    st.session_state.my_special_tab = "Edit Response"
                    b1, b2 = st.columns(2)
                    with b1:
                        if st.button("Show Response"):
                            st.session_state.my_special_tab = "Show Response"
                    with b2:
                        if b2.button("Edit Response"):
                            st.session_state.my_special_tab = "Edit Response"
    
    print(f'Value of execute_button_pressed: {st.session_state.execute_button_pressed}')
    print(f'Value of my_special_tab: {st.session_state.my_special_tab}')    
    if st.session_state.execute_button_pressed and st.session_state.my_special_tab == "Edit Response":
        with col2:
            
            print(f"Debug: st.session_state.result = {st.session_state.result}")  # Debug
            print(f"Debug: st.session_state.edited_result = {st.session_state.edited_result}")  # Debug
            
            if not 'store_button_clicked' in st.session_state:
                st.session_state.store_button_clicked = False
            
            if not 'edited_result' in st.session_state:
                st.session_state.edited_result = st.session_state.result
            
            with st.form(key=f"form_{tab}"):
                print('Inside the form')
                # Your existing widgets go here
                st.markdown("### Edit Response")
            
                edited_result = st.text_area(label="Edit AI Response:", value=st.session_state.edited_result, height=750, label_visibility="collapsed")

                if os.getenv("ES_INSTALLED") == "True":
                    # Set label store in memory
                    label = "Store Response in Search Memory"
                else:
                    label = "Generate Search Memory Record"
                
                feedback_options = ["none","accepted", "corrected","rejected"]
                feedback = st.selectbox("Revision Quality:", feedback_options)
                usernote = st.text_input("Revision Note:", "", max_chars=100)
                
                st.session_state.store_button_clicked = st.form_submit_button(label=label)
                
                if st.session_state.store_button_clicked:
                    print('Button clicked, do something')
                    st.session_state.edited_result = edited_result
                    st.session_state.usernote = usernote
                if not st.session_state.store_button_clicked:
                    print('Button NOT clicked, do something')
                    st.session_state.edited_result = st.session_state.result
            
            print("Before store_button condition")
            if st.session_state.store_button_clicked:
                print(">>>Inside store_button_clicked condition<<<")
                # st.session_state.my_special_tab = "Show Response"
                # Do something when the form is submitted
                final_data = {
                    "conversation_piece": st.session_state.conversation_piece,
                    "edited_response": edited_result,
                    "feedback": feedback,
                    "promptcategory": promptcategory,
                    "collection_id": selected_collection_id,
                    "usermessage_id": message_id,
                    "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "model": selected_model,
                    "userskillfocus": default_userskillfocus,
                    "prompt_tokens": st.session_state.prompt_tokens,
                    "completion_tokens": st.session_state.completion_tokens,
                    "prompt_cost": st.session_state.prompt_cost,
                    "completion_cost": st.session_state.completion_cost,
                    "total_cost": st.session_state.total_cost,
                    "username": st.session_state.username,
                    "usernote": usernote
                }
                
                # Calculate similarity score
                similarity_score = calculate_similarity(final_data["conversation_piece"]["model_response"], 
                                                    final_data["edited_response"])
                # Round similarity score to 2 decimal places
                similarity_score = round(similarity_score, 2)
                
                if similarity_score < 1:
                    final_data["feedback"] = "corrected"
                
                if similarity_score == 1:
                    final_data["feedback"] = "accepted"
                
                # Add similarity score to final_data
                final_data["similarityscore"] = similarity_score

                # Show the final data that will be stored in Elasticsearch
                st.write("JSON Object For Elasticsearch Store:")
                st.json(final_data)
                

                
                # Check if the es is connected and elasticsearch index exists
                if os.getenv("ES_INSTALLED") == "True" and not es.indices.exists(index=index_name):
                    # If the index does not exist, create it
                    mapping_schema = {
                        "settings": {
                            "analysis": {
                                "analyzer": {
                                    "custom_english_analyzer": {
                                        "type": "custom",
                                        "tokenizer": "standard",
                                        "filter": ["lowercase", "english_stemmer"]
                                    }
                                },
                                "filter": {
                                    "english_stemmer": {
                                        "type": "stemmer",
                                        "language": "english"
                                    }
                                }
                            }
                        },
                        "mappings": {
                            "properties": {
                                "conversation_piece": {
                                    "type": "nested",
                                    "properties": {
                                        "original_prompt": {
                                            "type": "nested",
                                            "properties": {
                                                "system": {"type": "text", "analyzer": "custom_english_analyzer"},
                                                "user": {"type": "text", "analyzer": "custom_english_analyzer"}
                                            }
                                        },
                                        "model_response": {"type": "text", "analyzer": "custom_english_analyzer"}
                                    }
                                },
                                "edited_response": {"type": "text", "analyzer": "custom_english_analyzer"},
                                "feedback": {"type": "keyword"},
                                "promptcategory": {"type": "keyword"},
                                "collection_id": {"type": "keyword"},
                                "usermessage_id": {"type": "long"},
                                "timestamp": {"type": "date"},
                                "model": {"type": "keyword"},
                                "similarityscore": {"type": "scaled_float", "scaling_factor": 100},
                                "userskillfocus": {"type": "keyword"},
                                "prompt_tokens": {"type": "integer"},
                                "completion_tokens": {"type": "integer"},
                                "prompt_cost": {"type": "scaled_float", "scaling_factor": 100},
                                "completion_cost": {"type": "scaled_float", "scaling_factor": 100},
                                "total_cost": {"type": "scaled_float", "scaling_factor": 100},
                                "username": {"type": "keyword"},
                                "usernote": {"type": "text", "analyzer": "custom_english_analyzer"},
                                "usertag": {"type": "keyword"}
                            }
                        }
                    }

                    es.indices.create(index=index_name, body=mapping_schema)
            
              
                # Store the document to elasticsearch
                try:
                    res = es.index(index=index_name, body=final_data)
                    if res['result'] == 'created' or res['result'] == 'updated':
                        st.sidebar.success(f"Document successfully stored in user index {index_name}.")
                    else:
                        st.sidebar.error("Document could not be written for an unknown reason.")
                except es_exceptions.ElasticsearchException as e:
                    st.sidebar.error(f"An error occurred: {e}")
            else:
                st.sidebar.error(f"Cannot Store Data, Elasticsearch is not connected.")
                
                # Reset the state
                # st.session_state.execute_button_pressed = False
                # st.session_state.result = ""
                # st.session_state.conversation_piece = ""
                
                # Show a success message
                # success_placeholder = st.empty()
                # success_placeholder.success("Action performed successfully!")
                
                # Make the success message disappear after 5 seconds
                # time.sleep(5)
                # success_placeholder.empty()
        print("After store_button condition")
        # Display the appropriate result in "Show Response"
    if st.session_state.my_special_tab == "Show Response" and (st.session_state.edited_result is not None or st.session_state.edited_result is not None):
        with col2:
            with st.container():
                # Display edited_result if it exists, otherwise display original result
                display_result = st.session_state.edited_result if st.session_state.edited_result else st.session_state.result
                st.markdown(display_result, unsafe_allow_html=True)            
