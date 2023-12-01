import numpy as np
import statistics
import logging
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain import PromptTemplate
from sklearn.cluster import KMeans
from tqdm import tqdm


def summarize(vectorstore, llm, embedding_dim=1536, max_tokens=10000, summary=True, keypoints=False, questions=False):
    index = vectorstore.store.index
    num_items = len(vectorstore.store.index_to_docstore_id)
    embedding_dim = embedding_dim
    vectors = []

    for i in range(num_items):
        vectors.append(index.reconstruct(i))

    embedding_matrix = np.array(vectors)
    doc_index = (vectorstore.store.docstore.__dict__['_dict'])
    chunk_tokens = []

    for key, value in doc_index.items():
        chunk_tokens.append(llm.model.get_num_tokens(value.page_content))

    mean_chunk_size = statistics.mean(chunk_tokens)
    target = max_tokens

    if target // mean_chunk_size <= len(chunk_tokens):
        num_clusters = (target // mean_chunk_size).__int__()
    else:
        num_clusters = len(chunk_tokens).__int__()

    logging.warning(f"Number of chunks chosen: {num_clusters}")
    print(f"Can afford {num_clusters} clusters , with mean chunk size of {mean_chunk_size} tokens, out of {len(chunk_tokens)} total chunks")

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit(embedding_matrix)

    closest_indices = []

    for i in range(num_clusters):
        distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
        closest_index = np.argmin(distances)
        closest_indices.append(closest_index)

    selected_indices = sorted(closest_indices)
    doc_ids = list(map(vectorstore.store.index_to_docstore_id.get, selected_indices))
    contents = list(map(vectorstore.store.docstore.__dict__['_dict'].get, doc_ids))

    map_prompt = """
    You will be given a single passage of a document. This section will be enclosed in triple backticks (```)
    Your goal is to identify what the passage tries to describe and give the general idea tha passage is discussing, as a summary. Do not focus on specific details and try to understand the general context. Start with This section is mainly obout,
    ```{text}```
    GENERAL IDEA:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
    map_chain = load_summarize_chain(llm=llm.model,
                                    chain_type="stuff",
                                    prompt=map_prompt_template)

    #Summary mappings
    print('Mapping summaries')
    
    results = []
    with tqdm(total=len(contents), desc="Processing documents") as pbar:
        for i in contents:
            res_2 = map_chain({"input_documents": [i]})['output_text']
            results.append(res_2)
            pbar.update(1)

    #results = [map_chain({"input_documents": [i]})['output_text'] for i in contents]
    summary_map = ''.join(['\n\nSummary: ' + s for s in results])
    summary_doc = Document(page_content = summary_map)

    summary_prompt = """
    You will be given a set of summaries of randomly selected passages from a document.
    Your goal is to write a paragraph on what the document is likely to be about.

    ```{text}```

    The document is:
    """

    insights_prompt = """
        You will be given a set of summaries of passages from a document.
        Your goal is to generate an overall general summary of the document using the summaries provided below within triple backticks.

        ```{text}```

        OVERALL CONTENT: Provide a list of bullet points.
        """

    questions_prompt = """
        You will be given a set of summaries of passages from a document.
        Your goal is to generate an overall general comprehensive summary of the document using the summaries provided below within triple backticks and ask them as questions.

        ```{text}```

        QUESTIONS: Provide a list of questions.
        """

    summary_prompt_template = PromptTemplate(template=summary_prompt, input_variables=["text"])
    insights_prompt_template = PromptTemplate(template=insights_prompt, input_variables=["text"])
    questions_prompt_template = PromptTemplate(template=questions_prompt, input_variables=["text"])

    summary_chain = load_summarize_chain(llm=llm.model,
                                    chain_type="stuff",
                                    prompt=summary_prompt_template)
    insights_chain = load_summarize_chain(llm=llm.model,
                                        chain_type="stuff",
                                        prompt=insights_prompt_template)
    questions_chain = load_summarize_chain(llm=llm.model,
                                        chain_type="stuff",
                                        prompt=questions_prompt_template)
    
    final_summary = None
    insights = None
    questions = None

    if(summary):
        final_summary = summary_chain({"input_documents": [summary_doc]})['output_text']
    if(keypoints):
        insights = insights_chain({"input_documents": [summary_doc]})['output_text']
    if(questions):
        questions = questions_chain({"input_documents": [summary_doc]})['output_text']

    out = {'summary':final_summary, 'keypoints':insights, 'questions':questions}
    return out


def keywords(vectorstore, llm, embedding_dim=1536, max_tokens=10000):
    index = vectorstore.store.index
    num_items = len(vectorstore.store.index_to_docstore_id)
    embedding_dim = embedding_dim
    vectors = []

    for i in range(num_items):
        vectors.append(index.reconstruct(i))

    embedding_matrix = np.array(vectors)
    doc_index = (vectorstore.store.docstore.__dict__['_dict'])
    chunk_tokens = []

    for key, value in doc_index.items():
        chunk_tokens.append(llm.model.get_num_tokens(value.page_content))

    mean_chunk_size = statistics.mean(chunk_tokens)
    target = max_tokens

    if target // mean_chunk_size <= len(chunk_tokens):
        num_clusters = (target // mean_chunk_size).__int__()
    else:
        num_clusters = len(chunk_tokens).__int__()

    print(f"Can afford {num_clusters} clusters , with mean chunk size of {mean_chunk_size} tokens, out of {len(chunk_tokens)} total chunks")

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit(embedding_matrix)

    closest_indices = []

    for i in range(num_clusters):
        distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
        closest_index = np.argmin(distances)
        closest_indices.append(closest_index)

    selected_indices = sorted(closest_indices)
    doc_ids = list(map(vectorstore.store.index_to_docstore_id.get, selected_indices))
    contents = list(map(vectorstore.store.docstore.__dict__['_dict'].get, doc_ids))

    keyword_prompt = """
    You will be given a single passage of a document. This section will be enclosed in triple backticks (```)
    Your goal is to identify what the passage tries to describe and give five comma separated un-numbered keywords from the passage.

    ```{text}```
    keywords:
    """
    keyword_prompt_template = PromptTemplate(template=keyword_prompt, input_variables=["text"])
    keyword_chain = load_summarize_chain(llm=llm.model,
                                    chain_type="stuff",
                                    prompt=keyword_prompt_template)

    #Summary mappings
    print('Mapping keywords')
    
    #res_2_key = [keyword_chain({"input_documents": [i]})['output_text'] for i in contents]
    res_2_key = []
    with tqdm(total=len(contents), desc="Processing documents") as pbar:
        for i in contents:
            res_2_key_t = keyword_chain({"input_documents": [i]})['output_text']
            res_2_key.append(res_2_key_t)
            pbar.update(1)
    
    #mapping keywords to chunks
    labels = kmeans.labels_
    string_list = res_2_key
    label_to_string = dict(zip(range(len(string_list)), string_list))
    mapped_strings = [label_to_string[label] for label in labels]
    return res_2_key