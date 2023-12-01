import asyncio
from asyncio import Lock
import altair as alt
import openai
import numpy as np
import logging
import os
import uuid
from tenacity import retry
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_extras.add_vertical_space import add_vertical_space
import plotly.graph_objects as go
import pandas as pd
import streamlit as st

openai.api_key = os.getenv("OPENAI_API_KEY", "Not found")
openai.organization = os.getenv("OPENAI_ORGANIZATION", "Not found")

def display_metrics_and_charts(uncertainty_scores, completions, n):
    structural_uncertainty = np.mean([np.mean(x) for x in uncertainty_scores['entropies']])
    conceptual_uncertainty = (0.5*uncertainty_scores['mean_choice_distance']) + (0.5*np.mean([np.mean(x) for x in uncertainty_scores['distances']]))

    st.text(f"Overall Cosine Distance Between Choices: {uncertainty_scores['mean_choice_distance']}")
    st.text(f"Overall Structural Uncertainty: {structural_uncertainty}")
    st.text(f"Overall Conceptual Uncertainty: {conceptual_uncertainty}")
    
    logger.debug(f"PROMPT: {user_input}\nSTRUCTURAL UNCERTAINTY: {structural_uncertainty}\nCONCEPTUAL UNCERTAINTY: {conceptual_uncertainty}")
    add_vertical_space(5)

    # Create columns for each choice
    cols = st.columns(n)
    
    for i in range(n):
        choice_text = completions['choices'][i]['text']
        entropies = uncertainty_scores['entropies'][i]
        distances = uncertainty_scores['distances'][i]
        logprobs = completions['choices'][i]['logprobs']['top_logprobs']
        mean_entropy = np.mean(entropies)
        mean_distance = np.mean(distances)

        tokens = completions['choices'][i]['logprobs']['tokens']

        if choice_text != ''.join(tokens):
            print("RESPONSE TEXT AND TOKENS DO NOT MATCH")
            print(f"Response Text: {choice_text}")
            print(f"Tokens: {''.join(tokens)}")

        # Fixed spacing between tokens
        fixed_spacing = 1

        # Calculate x positions based on fixed spacing
        x_positions = [0]
        for j in range(1, len(tokens)):
            x_positions.append(x_positions[-1] + len(tokens[j-1]) + fixed_spacing)

        # Create DataFrame for text and sparkline
        df = pd.DataFrame({
            'x': x_positions,
            'y_text': [1]*len(tokens),  # constant y-value to align text
            'y_entropy': [1.2 + entropy for entropy in entropies],
            'y_distance': [1.2 + dist for dist in distances],
            'tokens': tokens,
            'logprobs': ['\n'.join([f"{k}: {v}" for k, v in lp.items()]) for lp in logprobs],  # Format logprobs
            'entropy': entropies,
            'distance': distances,
        })

        # Create DataFrame for legend
        legend_data = pd.DataFrame({
            'x': [max(x_positions) - 15, max(x_positions) - 15],  # Position at the top-right
            'y': [1.8, 1.9],  # Adjust y-values to position legend
            'label': ['Entropy', 'Distance'],
            'color': ['blue', 'red']
        })

        # Create Altair chart for legend points
        legend_points = alt.Chart(legend_data).mark_point(
            size=100
        ).encode(
            x='x:Q',
            y='y:Q',
            color=alt.Color('color:N', legend=None, scale=alt.Scale(range=['blue', 'red']))  # Remove default legend
        )

        # Create Altair text for legend labels
        legend_text = alt.Chart(legend_data).mark_text(
            align='left',
            baseline='middle',
            dx=10,  # Nudges text to the right so it doesn't appear on top of the point
            color='white',
        ).encode(
            x='x:Q',
            y='y:Q',
            text='label',
        )

        with cols[i]:
            st.markdown(f"### Choice {i+1}")
            st.markdown(f"{choice_text}")
            add_vertical_space(2)
            st.markdown(f"**Mean Entropy:** {mean_entropy}")
            st.markdown(f"**Mean Distance:** {mean_distance}")

            # Create a selection object for panning
            pan = alt.selection_interval(encodings=['x'], bind='scales')

            # Create Altair chart
            text_chart = alt.Chart(df).mark_text(
                align='left',
                baseline='middle',
                color='white'  # Sets the text color to white
            ).encode(
                x=alt.X('x:Q', axis=None),  # Remove axis
                y=alt.Y('y_text:Q', axis=None, scale=alt.Scale(domain=[0, 2])),  # Remove axis
                text='tokens:N',
                tooltip=['logprobs', 'entropy', 'distance']
            ).add_params(pan)

            # Create Altair entropy sparkline chart
            entropy_chart = alt.Chart(df).mark_line(
                color='blue'
            ).encode(
            x='x:Q',
            y=alt.Y('y_entropy:Q', scale=alt.Scale(domain=[0, 1])),
            tooltip=alt.Tooltip('entropy:Q', title='Entropy Value')
            )

            # Add transparent points
            entropy_point_chart = alt.Chart(df).mark_point(
                color='blue',
                opacity=0  # Make points transparent
            ).encode(
                x='x:Q',
                y=alt.Y('y_entropy:Q', scale=alt.Scale(domain=[0, 1])),
                tooltip=alt.Tooltip('entropy:Q', title='Entropy Value')
            )

            # Create Altair distance sparkline chart
            distance_chart = alt.Chart(df).mark_line(
                color='red'
            ).encode(
            x='x:Q',
            y=alt.Y('y_distance:Q', scale=alt.Scale(domain=[0, 1])),
            tooltip=alt.Tooltip('distance:Q', title='Distance Value')
            )

            # Add transparent points
            distance_point_chart = alt.Chart(df).mark_point(
                color='red',
                opacity=0  # Make points transparent
            ).encode(
                x='x:Q',
                y=alt.Y('y_distance:Q', scale=alt.Scale(domain=[0, 1])),
                tooltip=alt.Tooltip('distance:Q', title='Distance Value')
            )

            # Combine line and points
            entropy_chart = entropy_chart + entropy_point_chart
            distance_chart = distance_chart + distance_point_chart

            # Layer the charts together
            combined_chart = alt.layer(text_chart, entropy_chart, distance_chart, legend_points, legend_text)

            # Display chart in Streamlit
            st.altair_chart(combined_chart, use_container_width=True)

async def update_status(status_text, message):
    status_text.text(message)
    await asyncio.sleep(0.1)  # Sleep to allow Streamlit to update the UI

@retry
async def get_embedding(input_text):
    response = await openai.Embedding.acreate(input=input_text, model="text-embedding-ada-002")
    return np.array(response['data'][0]['embedding'])

def calculate_normalized_entropy(logprobs):
    """
    Calculate the normalized entropy of a list of log probabilities.
    
    Parameters:
        logprobs (list): List of log probabilities.
        
    Returns:
        float: Normalized entropy.
    """

    # Calculate raw entropy
    entropy = -np.sum(np.exp(logprobs) * logprobs)
    
    # Calculate maximum possible entropy for N tokens sampled
    N = len(logprobs)
    max_entropy = np.log(N)
    
    # Normalize the entropy
    normalized_entropy = entropy/max_entropy
    return normalized_entropy

async def process_token_async(i, top_logprobs_list, choice, choice_embedding, max_tokens):
    """
    Asynchronously process a token to calculate its normalized entropy and mean cosine distance.
    
    Parameters:
        i (int): Token index.
        top_logprobs_list (list): List of top log probabilities for each token.
        choice (dict): The choice containing log probabilities and tokens.
        choice_embedding (array): Embedding of the full choice.
        max_tokens (int or None): Maximum number of tokens to consider for the partial string.
        
    Returns:
        tuple: Mean cosine distance and normalized entropy for the token.
    """
    top_logprobs = top_logprobs_list[i]
    normalized_entropy = calculate_normalized_entropy(list(top_logprobs.values()))
    
    tasks = []

    # Loop through each sampled token to construct partial strings and calculate embeddings
    for sampled_token in top_logprobs:
        tokens_to_use = choice['logprobs']['tokens'][:i] + [sampled_token]
        
        # Limit the number of tokens in the partial string if max_tokens is specified
        if max_tokens is not None and len(tokens_to_use) > max_tokens:
            tokens_to_use = tokens_to_use[-max_tokens:]
        constructed_string = ''.join(tokens_to_use)
        task = get_embedding(constructed_string)
        tasks.append(task)
        
    embeddings = await asyncio.gather(*tasks)
    
    cosine_distances = []

    # Calculate cosine distances between embeddings of partial strings and the full choice
    for new_embedding in embeddings:
        cosine_sim = cosine_similarity(new_embedding.reshape(1, -1), choice_embedding.reshape(1, -1))[0][0]
        cosine_distances.append(1 - cosine_sim)
        
    mean_distance = np.mean(cosine_distances)
    
    return mean_distance, normalized_entropy

async def calculate_uncertainty(response_object, max_tokens=None, status_text=None):
    """
    Asynchronously calculate uncertainty metrics for a given response object.
    
    Parameters:
        response_object (dict): The response object containing multiple choices.
        max_tokens (int or None): Maximum number of tokens to consider for the partial string.
        status_text (str or None): Optional status text for progress updates.
        
    Returns:
        dict: Dictionary containing lists of entropies, distances, and the mean choice-level distance.
    """

    # Initialize lists to store entropies, distances, and choice embeddings
    entropies = []
    distances = []
    choice_embeddings = []
    
    # Initialize counters and lock for task synchronization
    total_tasks = len(response_object['choices'])  # Total number of choices
    completed_tasks = 0  # Counter for completed tasks
    lock = Lock()  # Lock to synchronize updates to the counter and progress bar
    
    pbar = st.progress(0)  # Initialize Streamlit progress bar
    
    # Pre-calculate embeddings for each choice in the response object
    if status_text:
        await update_status(status_text, "Pre-calculating choice embeddings...")
    choice_embedding_tasks = [get_embedding(choice['text']) for choice in response_object['choices']]
    choice_embeddings = await asyncio.gather(*choice_embedding_tasks)
    
    async def process_choice(choice, choice_embedding):
        """
        Asynchronously process a single choice to calculate its mean cosine distances and normalized entropies.
        
        Parameters:
            choice (dict): The choice containing log probabilities and tokens.
            choice_embedding (array): Embedding of the full choice.
            
        Returns:
            tuple: Lists of mean cosine distances and normalized entropies for the choice.
        """

        nonlocal completed_tasks  # Declare as nonlocal to modify the shared counter

        top_logprobs_list = choice['logprobs']['top_logprobs']
        mean_cosine_distances = []
        normalized_entropies = []
        
        tasks = [process_token_async(i, top_logprobs_list, choice, choice_embedding, max_tokens) for i in range(len(top_logprobs_list))]
        results = await asyncio.gather(*tasks)
        
        for mean_distance, normalized_entropy in results:
            mean_cosine_distances.append(mean_distance)
            normalized_entropies.append(normalized_entropy)
        
        async with lock:  # Acquire lock to update shared state
            completed_tasks += 1  # Update the counter
            pbar.progress(completed_tasks / total_tasks)  # Update the progress bar
        
        return mean_cosine_distances, normalized_entropies
    
    if status_text:
        await update_status(status_text, "Processing choices... This may take a while depending on how many tokens are in each choice.")
    choice_tasks = [process_choice(choice, emb) for choice, emb in zip(response_object['choices'], choice_embeddings)]
    results = await asyncio.gather(*choice_tasks)
    
    if status_text:
        await update_status(status_text, "Calculating distances and entropies...")
    for mean_cosine_distances, normalized_entropies in results:
        distances.append(mean_cosine_distances)
        entropies.append(normalized_entropies) 
    
    # Calculate the mean cosine distance between all pairs of choices
    choice_distances = []
    for emb1, emb2 in combinations(choice_embeddings, 2):
        cosine_sim = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
        choice_distances.append(1 - cosine_sim)
    mean_choice_distance = np.mean(choice_distances)
    
    await update_status(status_text, "") 
    return {'entropies': entropies, 'distances': distances, 'mean_choice_distance': mean_choice_distance}

n = 5
n_logprobs = 5
max_tokens = 500

# Configure root logger to capture only WARN or higher level logs
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Configure logger to capture INFO-level logs
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Set page layout
st.set_page_config(layout='wide', page_title="LLM Uncertainty", page_icon=":mag:")

# Initialize session state for history if it doesn't exist
if 'history' not in st.session_state:
    st.session_state.history = []
if 'display_restored_data' not in st.session_state:
    st.session_state.display_restored_data = False
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""
if 'restore_clicks' not in st.session_state:
    st.session_state.restore_clicks = []

# Create columns for prompt input and history
col1, col2 = st.columns([2, 1])

with col1:
    st.title("Model Uncertainty Scoring")
    st.markdown(
    """
    This is a demo of several metrics that can be used to evaluate the uncertainty of an LLM's responses.

    You can read more details about it in [this blog post](https://www.watchful.io/blog/decoding-llm-uncertainties-for-better-predictability)
    
    * Conceptual uncertainty measures how sure the model is about what to say. 
    * Structural uncertainty measures how sure the model is about how to say it.

    Numbers closer to 1 indicate higher uncertainty, while highly certain outputs will be close to 0.

    You can interact with the charts that show up below - hover over each token to see the logprobs of the other top_n sampled tokens.
    """
    )
    user_input = st.text_area("Enter your prompt", value=st.session_state.user_input)
    status_text = st.empty()
    submit_button = st.button("Submit")

with col2:
    # Initialize History component
    st.markdown("## Recent Prompt History")
    for index, item in enumerate(st.session_state.history):
        unique_key = item['key']
        truncated_prompt = (item['prompt'][:50] + '...') if len(item['prompt']) > 50 else item['prompt']
        structural_uncertainty = np.mean([np.mean(x) for x in item['uncertainty_scores']['entropies']])
        conceptual_uncertainty = (0.5*item['uncertainty_scores']['mean_choice_distance']) + (0.5*np.mean([np.mean(x) for x in item['uncertainty_scores']['distances']]))
        mean_choice_distance = item['uncertainty_scores']['mean_choice_distance']
        
        # Create a row for each history element
        row_cols = st.columns([2, 1, 1, 1, 1])
        with row_cols[0]:
            st.write(f"{truncated_prompt}")
        with row_cols[1]:
            st.write(f"CD: {mean_choice_distance:.2f}")
        with row_cols[2]:
            st.write(f"SU: {structural_uncertainty:.2f}")
        with row_cols[3]:
            st.write(f"CU: {conceptual_uncertainty:.2f}")
        with row_cols[4]:
            st.session_state.restore_clicks[index] = st.button(f"Restore", key=f"restore_{unique_key}")
    
    for index, clicked in enumerate(st.session_state.restore_clicks):
        if clicked:
                st.session_state.display_restored_data = True
                st.session_state.user_input = st.session_state.history[index]['prompt']
                uncertainty_scores = st.session_state.history[index]['uncertainty_scores']
                completions = st.session_state.history[index]['completions']
                st.session_state.restore_clicks[index] = False


# Submit button right below the text box
if submit_button or st.session_state.display_restored_data:
    if submit_button:
        status_text.text("Creating Completions...")
        completions = openai.Completion.create(temperature=1,
                                           n=n,
                                           logprobs=n_logprobs,
                                           model="gpt-3.5-turbo-instruct", 
                                           max_tokens=max_tokens,
                                           prompt=user_input)
        uncertainty_scores = asyncio.run(calculate_uncertainty(completions, max_tokens=8191, status_text=status_text))
        unique_key = str(uuid.uuid4())   
        st.session_state.history.append({
            'key': unique_key,
            'prompt': user_input,
            'uncertainty_scores': uncertainty_scores,
            'completions': completions
        })
        st.session_state.restore_clicks.append(False)
        # Keep only the last 5 results
        if len(st.session_state.history) > 2:
            st.session_state.history.pop(0)
            st.session_state.restore_clicks.pop(0)

    display_metrics_and_charts(uncertainty_scores, completions, n) 
    st.session_state.display_restored_data = False