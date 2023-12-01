import box
import timeit
import yaml
import pprint

from langchain.pydantic_v1 import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain.chains.openai_functions import (
    create_structured_output_chain
)
from docs_qa.chains import build_llm
from github_qa.prompts import qa_template
from docs_qa.extract_search_terms import run_query_async
from github_qa.typesense_search import typesense_search_multiple
from typing import Sequence


pp = pprint.PrettyPrinter(indent=2)

# Import config vars
with open('github_qa/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


class RagContextRefs(BaseModel):
    # relevant_content: str = Field(..., description="Three or four sentences from the most relevant parts of the context document")
    source: str = Field(..., description="The metadata.source property")

class RagPromptReply(BaseModel):
    """Relevant context data"""
    helpful_answer: str = Field(..., description="The helpful answer")
    i_dont_know: bool = Field(..., description="True when unable to answer based on the given context.")
    relevant_contexts: Sequence[RagContextRefs] = Field(..., description="List of context documents that were relevant when answering the question.")



async def rag_with_typesense(user_input):

    durations = {
        'generate_searches': 0,
        'execute_searches': 0,
        'rag_query': 0,
        'total': 0
    }
    total_start = start = timeit.default_timer()
    extract_search_queries = await run_query_async(user_input)
    durations['generate_searches'] = timeit.default_timer() - start

    # print(f'generated queries:')
    # pp.pprint(extract_search_queries)

    start = timeit.default_timer()
    search_response = await typesense_search_multiple(extract_search_queries)
    durations['execute_searches'] = timeit.default_timer() - start

    # print(f'search response:')
    # pp.pprint(search_response)

    search_hits = [
        {
            'id': document['document']['id'],
            'url': document['document'].get('url', ''),
            'title': document['document'].get('title'),
            'body': document['document'].get('body')[:1000],
            'labels': document['document'].get('labels', []),
            'state': document['document'].get('state', ''),
            'closed_at': document['document'].get('closed_at'),
            'vector_distance': document.get('vector_distance', 1.0),
        }
        for result in search_response['results']
        if 'hits' in result
        for document in result['hits']
    ]    

    if len(search_hits) == 0:
        raise Exception("NoSearchResults")

    # print(f'All issues found:')
    # pp.pprint(search_hits)

    # need to preserve order in documents list
    # should only append doc if context is not too big
    loaded_docs = []
    loaded_source_urls = []
    doc_index = 0
    docs_length = 0

    while doc_index < len(search_hits):        
        search_hit = search_hits[doc_index]
        doc_index += 1      
        doc_trimmed = search_hit.get('body','')[:cfg.MAX_SOURCE_LENGTH]  

        loaded_doc = {
            'title': search_hit.get('title', ''),
            'labels': search_hit.get('labels', []),
            'state': search_hit.get('state', ''),
            'vector_distance': search_hit.get('vector_distance', 1.0),
            'page_content': doc_trimmed,
            'metadata': {            
                'source': search_hit.get('url', ''),                                
            }
        }    
        closed_at = search_hit.get('closed_at')
        if closed_at:
            loaded_doc['closed_at'] = closed_at

        # skip hits that are clearly out of range
        if search_hit.get('vector_distance', 1.0) > 0.9:
            continue

        if (docs_length + len(doc_trimmed)) > cfg.MAX_CONTEXT_LENGTH:
            break

        # limit result set length
        if len(loaded_docs) > 8:
            break

        docs_length += len(doc_trimmed)
        loaded_docs.append(loaded_doc) 
        loaded_source_urls.append(search_hit.get('url', ''))               


    print(f'Starting RAG structured output chain, llm: {cfg.MODEL_TYPE}')
    
    start = timeit.default_timer()
    llm = build_llm()
    prompt = ChatPromptTemplate.from_messages(
            [('system', 'You are a helpful assistant.'),
             ('human',  qa_template)]
        )

    runnable = create_structured_output_chain(RagPromptReply, llm, prompt)
    result = runnable.invoke({
        "context": yaml.dump(loaded_docs),
        "question": user_input
    })
    durations['rag_query'] = timeit.default_timer() - start
    durations['total'] = timeit.default_timer() - total_start

    # print(f"Time to run RAG structured output chain: {chain_end - chain_start} seconds")

    # print(f'runnable result:')
    # pp.pprint(result)

    if result['function'] is not None:
        relevant_sources = [{
            'url': context.source,
            'title': next((hit['title'] for hit in search_hits if hit['url'] == context.source), None),
        }
        for context in result['function'].relevant_contexts]
        rag_success = result['function'].i_dont_know != True
    else:
        relevant_sources = []

    response = {
        'result': result['function'].helpful_answer,        
        'rag_success': rag_success,
        'search_queries': extract_search_queries.searchQueries,
        'source_urls': loaded_source_urls,
        'source_documents': loaded_docs,
        'relevant_urls': relevant_sources,
        'durations': durations
    }

    # pp.pprint(response)

    return response