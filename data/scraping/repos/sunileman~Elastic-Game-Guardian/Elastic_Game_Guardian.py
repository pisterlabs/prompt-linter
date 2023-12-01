# Import the required libraries
import openai

from ingester import ingest_data_into_elastic
from utils.es_helper import create_es_client
from utils.openai_embedder import get_embedding
import streamlit as st
from utils.es_config import index_name
import pandas as pd
import time

from utils.query_builder import build_query
from variables import openai_api_type, openai_api_base, openai_api_version, es_cloudid, es_password, es_username, \
     openai_completion_deployment_name

openai.api_type = openai_api_type
openai.api_base = openai_api_base
openai.api_version = openai_api_version

try:

    username = es_username
    password = es_password
    cloudid = es_cloudid

    # get es object
    es = create_es_client(username, password, cloudid)

    print(es.info())
except Exception as e:
    print("Connection failed", e.errors)

# Set the GPT-3 API key
openai.api_key = st.secrets['pass']

# Create a list of numbers from 5 to 20
options = list(range(5, 21))

# Set the default index for the number 16
default_index = options.index(16)

st.image("./images/solution.png", width=300)

st.title('Elastic Game Guardian')
st.subheader('AI-Powered Age-Suitable Game Recommendations')

platform_selection = st.radio(
    "Gaming Platform:",
    ("Xbox", "PlayStation", "Nintendo", "All")
)

# Create a drop-down box with numbers from 5 to 20, defaulting to 12
ageSelection = st.selectbox('Age of Child:', options=options, index=default_index)

type_of_game_text = st.text_area("Game Theme")

# Mapping the platform_choice string to its corresponding platform value
platform_mapping = {
    "Xbox": "XBOX",
    "ps": "PS",
    "nintendo": "NINTENDO"
}

if st.button("Search", type='primary'):
    embeddings = get_embedding(type_of_game_text)

    query = build_query(embeddings, platform_selection)

    results = es.search(index=index_name, body=query)

    # Create a DataFrame to store the results
    df_output = pd.DataFrame(columns=["Name", "OpenAI Output"])

    num_results = min(5, len(results['hits']['hits']))

    stored_results = []

    for idx in range(num_results):
        name = results['hits']['hits'][idx]["_source"]["game_title"]
        platform = results['hits']['hits'][idx]["_source"]["platform"]
        response = openai.Completion.create(
            engine=openai_completion_deployment_name,
            prompt="Provide guidance on whether the Xbox game named '" + name + "' is appropriate for children aged " + str(
                ageSelection) + ". only provide definitive answers.  If uncertain, recommend no. Provide detailed explanation but limit to 40 words",
            max_tokens=128,
            temperature=0.5,
        )
        completion_output = response.choices[0].text.strip()
        stored_results.append((name, completion_output))
        time.sleep(4)

    for name, completion_output in stored_results:
        st.markdown(f"**Console: {platform}:Game Title:** {name}")
        st.markdown(f"**Guidance:** {completion_output}\n\n---\n")

# Create some space to push the button down
st.markdown('#')
st.markdown('#')
st.markdown('#')

if st.button('Ingest Game Data'):
    message = ingest_data_into_elastic()

    # Display the returned message in Streamlit
    st.text(message)
