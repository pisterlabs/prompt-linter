from dotenv import dotenv_values
config = dotenv_values()
import openai

# constants
EMBEDDING_MODEL = "text-embedding-ada-002"
openai.api_key = config.get("OPENAI_API_KEY")


from openai.embeddings_utils import (
    get_embedding,
    distances_from_embeddings,
    tsne_components_from_embeddings,
    chart_from_components,
    indices_of_nearest_neighbors_from_distances,
)


def get_embedding(text, model="text-embedding-ada-002"):
    """retrieve batch embeddings via api call"""
    #  text = text.replace("\n", " ")
    return openai.Embedding.create(input = text, model=model)['data']#[:]['embedding'] # returns [{'embedding}:[2,3], {..}, ..]

def embedding_from_string_list(
    string_list,
    model: str = EMBEDDING_MODEL
                                ) -> list:
    """Return embedding of given string in format [{'embedding}:[2,3], {..}, ..]"""
    # print("string_list", string_list)
    embeddings_raw = get_embedding(string_list, model)

    #post processing [{'embedding}:[2,3], {..}, ..] into multi-d list
    embeddings_list = []
    for i in range(len(embeddings_raw)):
        embeddings_list.append(embeddings_raw[i]['embedding'])
    return embeddings_list

def retrieve_rankings_per_string( text_block: str, # query data
                                        key_features, # list of strings; data quiried against
                                        k_nearest_neighbors: int = 2,
                                        model=EMBEDDING_MODEL,
                                        want_print=False
                                    ): #-> list[int]:
    
    original_data_string_list = [text_block] + key_features
    index_of_source_string = 0


    # get embeddings for all strings
    embeddings = embedding_from_string_list(original_data_string_list)
    
    # get the embedding of the source string
    query_embedding = embeddings[index_of_source_string]
    
    # get distances between the source embedding and other embeddings (function from embeddings_utils.py)
    distances = distances_from_embeddings(query_embedding, embeddings, distance_metric="cosine")
    
    # get indices of nearest neighbors (function from embeddings_utils.py)
    indices_of_nearest_neighbors = indices_of_nearest_neighbors_from_distances(distances)

    indices_of_nearest_neighbors = [int(n - 1) for n in indices_of_nearest_neighbors[1:]]

    # print out source string
    query_string = original_data_string_list[index_of_source_string]
    if want_print:
        print(f"Source string: {query_string}")
        # print out its k nearest neighbors
        k_counter = 0
        for i in indices_of_nearest_neighbors:
            # skip any strings that are identical matches to the starting string
            if query_string == original_data_string_list[i]:
                continue
            # stop after printing out k articles
            if k_counter >= k_nearest_neighbors:
                break
            k_counter += 1

            # print out the similar strings and their distances
            print(
                f"""
            --- Recommendation #{k_counter} (nearest neighbor {k_counter} of {k_nearest_neighbors}) ---
            String: {original_data_string_list[i]}
            Distance: {distances[i]:0.3f}"""
            )


    return indices_of_nearest_neighbors
