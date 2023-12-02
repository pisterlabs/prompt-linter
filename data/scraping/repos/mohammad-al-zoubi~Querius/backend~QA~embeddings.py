import json
import time
from tqdm import tqdm
import cohere
import hnswlib

Cohere_API_KEY = ""
co = cohere.Client(Cohere_API_KEY)

EMBEDDINGS_MODEL = "embed-multilingual-v2.0"
EMBEDDINGS_DIM = 768


def chunk_list(lst, chunk_size):
    """
    Chunk a list into sublists of specified size.
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def generate_chunk_log_embeddings(log_file_path, output_path_data='log_data.json',
                                  output_path_results='log_embeddings.json'):
    """
    Generates embeddings for each logline in logfile in form of:
    {"embeddings": [
        {"log_line": "log line 1", "embedding": [0.1, 0.2, 0.3, ...], "id": 0},
    ]}
    Args:
        log_file_path: path to the log_file.out

    Returns:

    """
    with open(log_file_path, 'r', encoding='utf-8') as file:
        logs = file.readlines()

    # Chunk the logs into 5000 log chunks
    chunk_size = 5000
    log_chunks = list(chunk_list(logs, chunk_size))

    log_data = {'lines': []}
    results = {'embeddings': []}
    start_time = time.time()

    for chunk_id, log_chunk in tqdm(enumerate(log_chunks), total=len(log_chunks)):
        # Sleep for 60 seconds to embed one chunk per minute
        elapsed_time = time.time() - start_time
        if chunk_id != 0:
            if elapsed_time < 60:
                time.sleep(60 - elapsed_time)

        start_time = time.time()
        embeds = co.embed(texts=log_chunk, model=EMBEDDINGS_MODEL, input_type='search_document').embeddings

        for i, embed in enumerate(embeds):
            log_data['lines'].append(
                {'log_line': log_chunk[i], 'id': i + chunk_id * chunk_size})
        results['embeddings'] += embeds

        # Save the results to a json file after processing each chunk
        with open(output_path_data, 'w', encoding='utf-8') as file:
            json.dump(log_data, file, ensure_ascii=False, indent=4)

        with open(output_path_results, 'w', encoding='utf-8') as file:
            json.dump(results, file, ensure_ascii=False, indent=4)

        # if chunk_id == len(log_chunks) - 1:
        #     break

    return log_data, results


# TODO: Deprecated. Remove this function after testing the new one.
def generate_log_embeddings(log_file_path, output_path='log_embeddings.json'):
    """
    Generates embeddings for each logline in logfile in form of:
    {"embeddings": [
        {"log_line": "log line 1", "embedding": [0.1, 0.2, 0.3, ...], "id": 0},
    ]}
    Args:
        log_file_path: path to the log_file.out

    Returns:

    """
    with open(log_file_path, 'r', encoding='utf-8') as file:
        logs = file.readlines()
    needed_logs = logs
    embeds = co.embed(texts=needed_logs, model=EMBEDDINGS_MODEL, input_type='search_document').embeddings
    results = {'embeddings': []}
    for i, embed in enumerate(embeds):
        results['embeddings'].append({'log_line': logs[i], 'embedding': embed, 'id': i})

    # Save the results to a json file
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)

    return embeds


def generate_query_embeddings(query):
    """
    Generates embeddings for a text query to be used for similarity search.

    Args:
        query [str]: query to be embedded

    Returns:
        embeds [list]: list of embeddings for the query

    """
    embeds = co.embed(texts=[query], model=EMBEDDINGS_MODEL, input_type='search_document').embeddings
    return embeds


def load_embeddings(path_to_log_embeddings, mode='list'):
    """
    Loads the embeddings from a json file.

    Args:
        path_to_embeddings [str]: path to the json file containing the embeddings

    Returns:
        doc_embs [list]: list of embeddings for the documents
        query_embs [list]: list of embeddings for the query

    """
    with open(path_to_log_embeddings, 'r', encoding='utf-8') as file:
        data = json.load(file)

    if mode == 'json':
        return data['lines']
    elif mode == 'list':
        return data['embeddings']


def rerank_results(query, log_embeddings, log_jsons, index, top_n=10):
    """
    Ranks the results of the similarity search based on the similarity score.
    """

    query_emb = generate_query_embeddings(query)
    start = time.time()
    k = len(log_embeddings) if len(log_embeddings) < 5000 else 1000
    doc_ids = index.knn_query(query_emb, k=k)[0][0]
    end = time.time()
    print(f"Search time: {end - start}")

    results = []
    for doc_id in tqdm(doc_ids, total=len(doc_ids)):
        results.append({"text": log_jsons[doc_id]['log_line'], "id": int(doc_id)})

    start = time.time()
    rerank_results = co.rerank(query=query, documents=results, top_n=top_n, model='rerank-multilingual-v2.0',
                               max_chunks_per_doc=1)  # Change top_n to change the number of results returned. If top_n is not passed, all results will be returned.
    end = time.time()
    print(f"Reranking time: {end - start}")

    for result in rerank_results:
        print(result.document['text'])
    return rerank_results


def create_search_index(log_embeddings):
    start = time.time()
    # Create a search index
    index = hnswlib.Index(space='ip', dim=EMBEDDINGS_DIM)
    index.init_index(max_elements=len(log_embeddings), ef_construction=512, M=64)
    index.add_items(log_embeddings, list(range(len(log_embeddings))))
    end = time.time()
    print(f"Indexing time: {end - start}")
    return index


if __name__ == '__main__':
    path_to_logs = r"C:\Users\Mohammad.Al-zoubi\Documents\projects\Querius\backend\QA\data\test_log_30k.out"
    output_path = r"C:\Users\Mohammad.Al-zoubi\Documents\projects\Querius\backend\QA\log_embeddings_multi_v2_3111k.json"
    query = "What error messages are there?"

    # generate_chunk_log_embeddings(path_to_logs,
    #                               output_path_data='data/log_data_multi_30k.json',
    #                               output_path_results='data/log_embeddings_multi_30k.json')
    # generate_query_embeddings(query)
    # log_embeddings = load_embeddings(output_path)
    # log_jsons = load_embeddings(output_path, mode='json')
    # rank_results(query, log_embeddings, log_jsons)
