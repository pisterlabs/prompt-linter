from deta import Base
from typing import List
import logging
from urllib.parse import urlparse
from jsonformatter import JsonFormatter
from newspaper import Article

import traceback

import cohere
from annoy import AnnoyIndex

format = '''{
    "level":           "levelname",
    "logger_name":     "%(name)s.%(funcName)s",
    "timestamp":       "asctime",
    "message":         "message"
}'''

def get_logger(name: str, level: int = logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = JsonFormatter(format)

    logHandler = logging.StreamHandler()
    logHandler.setFormatter(formatter)
    logHandler.setLevel(level)

    logger.addHandler(logHandler)
    return logger

def parse_website(url: str):
    """
    Get HTML text from URL
    """
    article_report = {
        'authors': None,
        'published_date': None,
        'article_title': None
    }
    article = Article(url, language='en')
    article.download()
    try:
        article.parse()
    except:
        return article_report, None
    
    article_report['authors'] = article.authors
    article_report['published_date'] = article.publish_date.strftime('%d/%m/%Y')
    article_report['article_title'] = article.title
    return article_report, article.text

#
# Cohere logic to identify, search and verify claims
#

def get_claims_form_text(logger, co: cohere.client.Client, article_text: str, article_info: dict):
    """
    Get claims from HTML text
    """
    #claims = [claim for claim in article_text.strip().split("\n") if len(claim)>0]
    #return claims
    full_text = 'Article: ' + article_info['article_title'] + ' ' + article_text[:5000]  + '\n'*2 + 'Claim: '
    try:
        response = co.generate(
            model='aa42cd3e-3154-4904-9c79-f2cbea2a3c74-ft',
            prompt=full_text
        )
    except Exception as e:
        logger.info(f"Error: {str(e)}")
    return [response.generations[0].text]

def get_claim_embeddings(co: cohere.client.Client, claims: List[str]):
    return co.embed(
        texts=claims,
        model="large",
        truncate="LEFT"
    ).embeddings

def _search_claim_at_index(
    logger,
    search_index: AnnoyIndex,
    db: Base,
    embedding: List[float],
    top_k: int = 10,
):
    """
    Search a claim in the embedding index and return top_k most similar claims
    """
    
    try:
        # Retrieve the nearest neighbors
        similar_claims = search_index.get_nns_by_vector(
            embedding,
            top_k,
            include_distances=True
        )
    except Exception as error:
        just_the_string = traceback.format_exc()
        logger.info(just_the_string)

    claims_with_score = []

    for idx, score in zip(similar_claims[0], similar_claims[1]):
        claim_obj = db.get(str(idx))
        if claim_obj:
            claim_obj['score'] = score
        claims_with_score.append(claim_obj)

    return claims_with_score

def _calculate_match(claim: str, closest_claims: List[dict]):
    # Check if claim is found in DB
    if claim == closest_claims[0]['claim_text'] or closest_claims[0]['score'] < 0.3:
        return {
            'predicted_category': closest_claims[0]['category'],
            'found': True,
            'claim_text': claim,
        }
    return {
        'found': False
    }

def verify_claims(
    logger,
    search_index: AnnoyIndex,
    db: Base,
    embeddings: List[List[float]],
    claims: List[str],
    top_k: int = 10,
):
    """
    Search each claim in index and determine if there's exact match.
    Any found claim is returned along with the category assigned.
    """
    found_claims = []
    for emb, claim in zip(embeddings, claims):
        logger.info(f"searching claim: {claim}")
        most_similar_emb_objs = _search_claim_at_index(logger, search_index, db, emb, top_k)
        logger.info(f"Result found: {most_similar_emb_objs}")
        match_result = _calculate_match(claim, most_similar_emb_objs)
        if match_result['found']:
            found_claims.append(match_result)
    return found_claims