import json
import itertools
import openai
import pinecone
import psycopg2
from google.cloud import secretmanager


client = secretmanager.SecretManagerServiceClient()
name = f"projects/vidbox-7d2c1/secrets/pinecone-api-key/versions/latest"
response = client.access_secret_version(request={"name": name})

pinecone.init(
    api_key=response.payload.data.decode("UTF-8"),
    environment='us-west1-gcp-free'
)
def filter_keys(dictionary, keys_to_keep):
    """
    Filters the dictionary to only keep keys in keys_to_keep.
    """
    return {k: dictionary[k] for k in keys_to_keep if k in dictionary}

def get_metadata():
    conn = psycopg2.connect(
        database="vidbox-backend_development",
        host="localhost",
        user="vishaalganesan",
        password="vish",
        port="5433")

    # Create a cursor to execute SQL commands.
    cursor = conn.cursor()

    # Execute the SQL command to select all rows from the 'movie_infos_top_rated' table.
    cursor.execute("SELECT * FROM movie_infos_top_rated")
    results = cursor.fetchall()

    # Fetch the column names.
    columns = [desc[0] for desc in cursor.description]

    print("Processing started...")
    all_metadata = dict()
    for idx, row in enumerate(results):
        row_metadata = dict()
        curr_id = ""
        for column_name, value in zip(columns, row):
            if column_name == "id":
                curr_id = value
            if column_name == "release_date": value = str(value)
            row_metadata[column_name] = value
        row_metadata = filter_keys(row_metadata, ['plot', "title", "overview"])
        all_metadata[curr_id] = row_metadata
    cursor.close()
    conn.close()
    return all_metadata
def read_and_transform_data(file_path):
    print("Reading and transforming data...")
    metadata = get_metadata()
    with open(file_path, 'r') as file:
        data = json.load(file)
    vectors_to_insert = []
    for key, value in data.items():
        vector = value[0]['embedding']
        if len(vector) != 1536:
            print(f"Warning: Embedding with key {key} has {len(vector)} dimensions instead of 1536.")
        vectors_to_insert.append((key, vector, metadata[int(key)]))
    print(f"Total vectors to insert: {len(vectors_to_insert)}")
    return vectors_to_insert
def chunks(iterable, batch_size=100):
    print(f"Dividing data into chunks of size {batch_size}...")
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))

def upsert():
    # Read and transform the data
    file_path = "embeddings2.json"
    vectors_to_insert = read_and_transform_data(file_path)

    # Define batch size
    batch_size = 100

    # Connect to the Pinecone index with parallel processing enabled
    index_name = "vidbox-movie-search-index"
    print(f"Connecting to Pinecone index {index_name}...")

    # Initialize a file to store IDs of vectors that were too large
    with open('failed_vector_ids.txt', 'w') as failed_ids_file:

        with pinecone.Index(index_name, pool_threads=30) as index:
            print("Sending upsert requests in parallel with batching...")

            for ids_vectors_chunk in chunks(vectors_to_insert, batch_size=batch_size):
                try:
                    async_result = index.upsert(vectors=ids_vectors_chunk, async_req=True)
                    async_result.get()
                except Exception as e:
                    print(f"Caught Exception: {e}")

                    # Log the exception and retry with empty plot
                    for failed_id, vector, metadata in ids_vectors_chunk:
                        failed_ids_file.write(f"{failed_id}\n")

                        # Update the metadata to have an empty plot and retry
                        metadata['plot'] = ""
                        try:
                            async_result = index.upsert(vectors=[(failed_id, vector, metadata)], async_req=True)
                            async_result.get()
                        except Exception as inner_e:
                            print(f"Failed to retry upsert for id {failed_id}: {inner_e}")

                    continue  # Skip this iteration and continue with the next chunk

            print("Waiting for responses...")
            print("Upsert operation completed.")

    print("Done.")
def query():
    name1 = f"projects/vidbox-7d2c1/secrets/openai_secret/versions/latest"
    response1 = client.access_secret_version(request={"name": name1})
    openai.api_key = response1.payload.data.decode("UTF-8")

    pinecone.init(
        api_key=response.payload.data.decode("UTF-8"),
        environment='us-west1-gcp-free'
    )

    index = pinecone.Index('vidbox-movie-search')
    query_vector = openai.Embedding.create(
        model="text-embedding-ada-002",
        input="the lord of the rings" # Use the concatenated text as input
    )
    embedding = query_vector['data'][0]['embedding']
    print(embedding)
    return embedding


#query()
# x = get_metadata()
# print(x[20069])
upsert()
#print(get_metadata())
