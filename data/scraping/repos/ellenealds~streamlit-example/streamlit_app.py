import streamlit as st
import pandas as pd
import cohere
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.decomposition import PCA

# create a section where the user can input a reflection
st.header('Reflections')
reflection = st.text_area('Enter your reflection here')

# store a dataset in the backend from a csv file
df = pd.read_csv('ksbs_embeds.csv')

# Reduce dimensionality using PCA


# Function to return the principal components
def get_pc(arr,n):
    pca = PCA(n_components=n)
    embeds_transform = pca.fit_transform(arr)
    return embeds_transform


sample = 65
# Reduce embeddings to 2 principal components to aid visualization
embeds = np.array(df['ksbs_embeds'].tolist())
embeds_pc2 = get_pc(embeds,2)
# Add the principal components to dataframe
df_pc2 = pd.concat([df, pd.DataFrame(embeds_pc2)], axis=1)

# Plot the 2D embeddings on a chart
df_pc2.columns = df_pc2.columns.astype(str)


co = cohere.Client(api_key='ccg4UEstp9A7VdFKhZQvlShYDU5BAaIUuSnuTluY')
# Get text embeddings
def get_embeddings(text,model='medium'):
    output = co.embed(
                    model=model,
                    texts=[text])
    return output.embeddings[0]


def get_similarity(target,candidates):
    # Turn list into array
    candidates = np.array(candidates)
    target = np.expand_dims(np.array(target),axis=0)

    # Calculate cosine similarity
    sim = cosine_similarity(target,candidates)
    sim = np.squeeze(sim).tolist()
    sort_index = np.argsort(sim)[::-1]
    sort_score = [sim[i] for i in sort_index]
    similarity_scores = zip(sort_index,sort_score)

    # Return similarity scores
    return similarity_scores

# create a function that takes a query and returns the top 5 similar queries a dataframe will be created with a row per query and a column for the similarity score
def get_similar_queries(query):
    # Get embeddings of the new query
    new_query_embeds = get_embeddings(query)
    
    # Get the similarity between the search query and existing queries
    similarity = get_similarity(new_query_embeds,embeds[:sample])
    
    # Create a dataframe with the similarity scores
    df_sim = pd.DataFrame(similarity,columns=['index','similarity'])
    df_sim = df_sim.merge(df,left_on='index',right_index=True)
    # append the query to the dataframe
    df_sim['query'] = query
    
    return df_sim

# create a button that runs function to get similar queries from the reflections
if st.button('Get similar reflections'):
    # get the similar queries
    similar_queries = get_similar_queries(reflection)
    # display the similar queries
    st.write(similar_queries)
