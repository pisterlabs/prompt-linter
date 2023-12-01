import os
import sys
import time
from argparse import Namespace
from pathlib import Path
import openai
import pandas as pd
from AbstractClusterBERTUtility import AbstractClusterBERTUtility
from KeyWordExtractionUtility import KeywordExtractionUtility
from stanza.server import CoreNLPClient
from tenacity import retry, wait_random_exponential, stop_after_attempt
# Set Sentence Transformer path
# sentence_transformers_path = os.path.join('/Scratch', getpass.getuser(), 'SentenceTransformer')
# if os.name == 'nt':
#     sentence_transformers_path = os.path.join("C:", os.sep, "Users", getpass.getuser(), "SentenceTransformer")
# Path(sentence_transformers_path).mkdir(parents=True, exist_ok=True)
# model_name = "all-mpnet-base-v2"
# device = 'cpu'
# model = SentenceTransformer(model_name, cache_folder=sentence_transformers_path, device='cpu')

# GPT API setup
openai.organization = "org-yZnUvR0z247w0HQoS6bMJ0WI"
openai.api_key = os.getenv("OPENAI_API_KEY")


class KeywordExtraction:
    # def __init__(self, _cluster_no):
    def __init__(self):
        self.args = Namespace(
            case_name='AIMLUrbanStudyCorpus',
            embedding_name='OpenAIEmbedding',
            model_name='curie',
            phase='keyword_extraction_phase',
            previous_phase='abstract_clustering_phase',
            path='data',
            diversity=0.5
        )
        # # Use the GPT model to find top 5 relevant and less redundant keywords from each abstract
        # Load the results from previous phase
        path = os.path.join('output', self.args.case_name + '_' + self.args.embedding_name, self.args.previous_phase,
                            self.args.case_name + '_clusters.json')
        self.corpus_docs = pd.read_json(path).to_dict("records")
        # Loaded the cluster results
        path = os.path.join('output', self.args.case_name + '_' + self.args.embedding_name, self.args.previous_phase,
                            self.args.case_name + '_cluster_terms.json')
        cluster_df = pd.read_json(path)
        folder = os.path.join('output', self.args.case_name + '_' + self.args.embedding_name,
                              self.args.phase)
        path = os.path.join(folder, self.args.case_name + '_cluster_terms.csv')
        cluster_df.to_csv(path, encoding='utf-8', index=False)
        path = os.path.join(folder, self.args.case_name + '_cluster_terms.json')
        cluster_df.to_json(path, orient='records')
        # Output cluster terms to this phase
        self.clusters = cluster_df.to_dict("records")
        # print(self.clusters)

    def extract_doc_key_phrases_by_similarity_diversity(self):
        @retry(wait=wait_random_exponential(min=30, max=60), stop=stop_after_attempt(6))
        def get_embedding(text: str, engine="text-similarity-" + self.args.model_name + "-001"):
            # replace newlines, which can negatively affect performance.
            text = text.replace("\n", " ")
            return openai.Embedding.create(input=[text], engine=engine)["data"][0]["embedding"]
        try:
            folder = os.path.join('output', self.args.case_name + '_' + self.args.embedding_name,
                                  self.args.phase)
            Path(folder).mkdir(parents=True, exist_ok=True)
            # Load doc vectors from GPT
            path = os.path.join(folder, 'doc_vectors', 'doc_vectors.json')
            doc_vectors = pd.read_json(path).to_dict("records")
            # Save/load all candidate vectors
            candidate_vector_folder = os.path.join('output', self.args.case_name + '_' + self.args.embedding_name,
                                                   self.args.phase, 'candidate_vectors')
            path = os.path.join(candidate_vector_folder, 'candidate_vectors.json')
            Path(candidate_vector_folder).mkdir(parents=True, exist_ok=True)
            if os.path.exists(path):
                candidate_vector_results = pd.read_json(path, compression='gzip').to_dict("records")
            else:
                candidate_vector_results = list()
            # Collect collocation phrases from each cluster of articles
            with CoreNLPClient(
                    annotators=['tokenize', 'ssplit', 'pos'],
                    timeout=30000,
                    memory='6G') as client:
                # cluster_no_list = [8]
                for cluster_result in self.clusters:
                    cluster_no = cluster_result['cluster']
                    cluster_docs = list(filter(lambda d: d['Cluster'] == cluster_no, self.corpus_docs))
                    results = list()  # Store the keywords (candidate words) for all the abstracts in a cluster
                    for doc in cluster_docs:
                        doc_id = doc['DocId']
                        # Get the first doc
                        doc = next(doc for doc in cluster_docs if doc['DocId'] == doc_id)
                        doc_text = AbstractClusterBERTUtility.preprocess_text(doc['Abstract'])
                        doc_vector = next(d['DocVectors'] for d in doc_vectors if d['DocId'] == doc_id)
                        # End of for loop
                        try:
                            # Collect all the candidate collocation words
                            candidates = KeywordExtractionUtility.generate_collocation_candidates(doc_text, client)
                            # Collect and cache all the vectors of candidate words
                            candidate_vectors = list()
                            for candidate in candidates:
                                # Check if the candidate vector appear before
                                found = next((r for r in candidate_vector_results if r['candidate'].lower() == candidate.lower()), None)
                                if found:
                                    candidate_vector = found['vector']
                                else:
                                    candidate_vector = get_embedding(candidate)
                                    candidate_vector_results.append({'candidate': candidate.lower(),
                                                                     'vector': candidate_vector})
                                candidate_vectors.append(candidate_vector)
                            assert len(candidates) == len(candidate_vectors)
                            # Compute the similarities between candidate words and abstract using GPT
                            candidate_scores = KeywordExtractionUtility.compute_similar_score_key_phrases_GPT(doc_vector,
                                                                                                              candidates,
                                                                                                              candidate_vectors)
                            # print(", ".join(n_gram_candidates))
                            # candidate_scores = KeywordExtractionUtility.compute_similar_score_key_phrases(model,
                            #                                                                               doc_text,
                            #                                                                               n_gram_candidates)
                            # candidate_similar_scores = KeywordExtractionUtility.sort_candidates_by_similar_score(candidate_scores)
                            # Rank the high scoring phrases
                            mmr_keywords_scores = KeywordExtractionUtility.re_rank_phrases_by_maximal_margin_relevance(
                                 doc_vector, candidates, candidate_vectors, self.args.diversity)
                            mmr_keywords = list(map(lambda p: p['keyword'], mmr_keywords_scores))
                            # Obtain top five key phrases
                            result = {'cluster': cluster_no, 'doc_id': doc_id,
                                      'keywords': mmr_keywords[:5],
                                      'candidates': candidate_scores}
                            results.append(result)
                            print("Complete to extract the key phrases from document {d_id}".format(d_id=doc_id))
                        except Exception as __err:
                            print("Error occurred! {err}".format(err=__err))
                            sys.exit(-1)
                    # Write the candidate vectors to JSON file
                    path = os.path.join('output', self.args.case_name + '_' + self.args.embedding_name,
                                        self.args.phase, 'candidate_vectors', 'candidate_vectors.json')
                    candidate_vector_df = pd.DataFrame(candidate_vector_results)
                    candidate_vector_df.to_json(path, orient='records', compression='gzip')
                    # # Write key phrases to csv file
                    df = pd.DataFrame(results)
                    doc_keyword_folder = os.path.join(folder, 'doc_keywords')
                    # Map the list of key phrases (dict) to a list of strings
                    Path(doc_keyword_folder).mkdir(parents=True, exist_ok=True)
                    path = os.path.join(doc_keyword_folder, 'doc_keyword_cluster_#' + str(cluster_no) + '.csv')
                    df.to_csv(path, encoding='utf-8', index=False)
                    path = os.path.join(doc_keyword_folder, 'doc_keyword_cluster_#' + str(cluster_no) + '.json')
                    df.to_json(path, orient='records')
                    print("Output the keywords for the docs in cluster #" + str(cluster_no))
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Combine doc keywords
    def output_doc_keywords(self):
        try:
            folder = os.path.join('output', self.args.case_name + '_' + self.args.embedding_name,
                                  self.args.phase)
            doc_keyword_folder = os.path.join(folder, 'doc_keywords')
            # Load candidate word vectors
            path = os.path.join(folder, 'candidate_vectors', 'candidate_vectors.json')
            candidate_vectors = pd.read_json(path, compression='gzip').to_dict("records")
            # print(candidate_vectors)
            # Collect keyword vectors
            keyword_vectors = list()
            # Combine the keywords of all abstracts to corpus
            results = list()
            for cluster in self.clusters:
                cluster_id = cluster['cluster']
                # Get key phrases of abstracts in a cluster
                path = os.path.join(doc_keyword_folder, 'doc_keyword_cluster_#{c}.json'.format(c=cluster_id))
                doc_keywords = pd.read_json(path).to_dict("records")
                for doc_keyword in doc_keywords:
                    doc_id = doc_keyword['doc_id']
                    candidates = doc_keyword['candidates']
                    keywords = doc_keyword['keywords']
                    # Retrieve and store keyword vectors
                    for keyword in keywords:
                        # Check if keyword vector exists
                        found = next((vector for vector in keyword_vectors if vector['keyword'].lower() == keyword), None)
                        if not found:
                            keyword_vector = next((vector['vector'] for vector in candidate_vectors
                                                  if vector['candidate'].lower() == keyword.lower()), None)
                            assert keyword_vector is not None
                            keyword_vectors.append({"keyword": keyword.lower(), "vector": keyword_vector})
                    # Include candidate words and keywords to each abstract
                    doc = next(doc for doc in self.corpus_docs if doc['DocId'] == doc_id)
                    doc['CandidateWords'] = candidates
                    doc['GPTKeywords'] = keywords
                    results.append(doc)
            # Output corpus doc (with CandidateWords and Keywords using GTP model)
            df = pd.DataFrame(results, columns=[
                'Cluster', 'DocId', 'GPTKeywords', 'CandidateWords', 'Title', 'Abstract',
                'Cited by', 'Author Keywords', 'Year', 'Source title', 'Authors', 'DOI',
                'Document', 'Type', 'x', 'y'
            ])
            path = os.path.join(folder, self.args.case_name + '_clusters.csv')
            df.to_csv(path, index=False, encoding='utf-8')
            path = os.path.join(folder, self.args.case_name + '_clusters.json')
            df.to_json(path, orient='records')
            print('Output key phrases per doc to ' + path)
            # Output keyword vectors
            keyword_vector_df = pd.DataFrame(keyword_vectors)
            path = os.path.join(folder, 'keyword_vectors.json')
            keyword_vector_df.to_json(path, orient='records', compression='gzip')
        except Exception as err:
            print("Error occurred! {err}".format(err=err))


# Main entry
if __name__ == '__main__':
    try:
        kp = KeywordExtraction()
        # Extract keyword for each article
        # kp.extract_doc_key_phrases_by_similarity_diversity()
        kp.output_doc_keywords()
    except Exception as err:
        print("Error occurred! {err}".format(err=err))
