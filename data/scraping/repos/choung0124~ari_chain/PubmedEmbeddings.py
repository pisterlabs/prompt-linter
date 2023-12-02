import psycopg2
from sentence_transformers import CrossEncoder
import itertools
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from psycopg2 import connect

class PubmedSearchEngine:
    def __init__(self, host="localhost", port="5432", dbname='pubmed', user="hschoung", password="Reeds0124"):
        self.conn = connect(dbname=dbname, user=user, password=password, host=host, port=port)
        self.encoder = CrossEncoder('pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb')
        self.query_embedding_model = HuggingFaceEmbeddings(
            model_name='pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb',
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def rerank(self, query, results):
        scores = self.encoder.predict([(query, item[3]) for item in results])
        sorted_results = [(score, *result) for score, result in sorted(zip(scores, results), reverse=True)]
        return sorted_results[:1]

    def hybrid_search(self, query):
        query_vector = self.query_embedding_model.embed_query(query)
        with self.conn.cursor() as c:
            c.execute("""
                SELECT pmid, title, doi, abstract
                FROM pubmed_articles
                WHERE (pmid IS NOT NULL AND pmid <> '')
                AND (title IS NOT NULL AND title <> '')
                AND (doi IS NOT NULL AND doi <> '')
                AND (abstract IS NOT NULL AND abstract <> '')
                AND to_tsvector('english', title || ' ' || abstract) @@ plainto_tsquery('english', %s)
                ORDER BY pubmed_bert_vectors <#> %s::vector DESC, ts_rank_cd(to_tsvector('english', title || ' ' || abstract), plainto_tsquery('english', %s)) DESC
                LIMIT 1000
            """, (query, query_vector, query))
            results = c.fetchall()
        return results

    def query(self, question):
        results = self.hybrid_search(question)
        if results:
            results = self.rerank(question, results)
            formatted_results = []
            for score, abstract_id, title, doi, abstract_text in results:
                # Remove multiple newlines from the abstract
                abstract_text = abstract_text.replace("\n\n", "\n").replace("\n\n\n", "\n")
                # Build the formatted string
                formatted_result = f"Article Title: {title}\nArticle PMID: {abstract_id}\nAbstract: {abstract_text}\nScore: {score}\n"
                formatted_results.append(formatted_result)
            # Join all results into a single string separated by a newline
            return "\n".join(formatted_results)
        else:
            return "No results found."
        
def get_abstracts(search_engine, queries):
    abstracts = []
    for query in queries:
        result = search_engine.query(query)
        if result != "No results found.":
            # Split the result string by newline and get the abstract part
            result_parts = result.split("\n")
            abstract = result_parts[2] # The abstract is the third part
            # Remove the "Abstract: " prefix and add it to the list
            abstracts.append(abstract[10:])
    # Join all abstracts into a single string separated by a newline
    return "\n".join(abstracts)