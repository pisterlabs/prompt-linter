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
from docs_qa.prompts import qa_template
from docs_qa.extract_search_terms import run_query_async
import docs_qa.typesense_search as search
from typing import Sequence


pp = pprint.PrettyPrinter(indent=2)

# Import config vars
with open('docs_qa/config/config.yml', 'r', encoding='utf8') as ymlfile:
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
        'phrase_similarity_search': 0,
        'rag_query': 0,
        'total': 0
    }
    total_start = start = timeit.default_timer()
    extract_search_queries = await run_query_async(user_input)
    durations['generate_searches'] = timeit.default_timer() - start

    print(f'Query language code: {extract_search_queries.userInputLanguage}')

    # print(f'generated queries:')
    # pp.pprint(extract_search_queries)

    # start = timeit.default_timer()
    # search_response = await search.typesense_search_multiple(extract_search_queries)
    # durations['execute_searches'] = timeit.default_timer() - start

    
    start = timeit.default_timer()
    search_phrase_hits = await search.lookup_search_phrases_similar(extract_search_queries)
    durations['phrase_similarity_search'] = timeit.default_timer() - start

    print(f'url list:')
    pp.pprint(search_phrase_hits)

    start = timeit.default_timer()
    search_response = await search.typesense_retrieve_all_by_url(search_phrase_hits)
    durations['execute_searches'] = timeit.default_timer() - start

    search_hits = [
        {
            'id': document['document']['id'],
            'url': document['document']['url_without_anchor'],
            'lvl0': document['document']['hierarchy.lvl0'],
            'content_markdown': document['document']['content_markdown'],
        }
        for result in search_response['results']
        for hit in result['grouped_hits']
        for document in hit['hits']
    ]    

    # print(f'All source document urls:')
    # pp.pprint(search_hits)

    start = timeit.default_timer()

    loaded_docs = []
    loaded_urls = []
    loaded_search_hits = []
    doc_index = 0
    docs_length = 0

    # need to preserve order in documents list
    # should only append doc if context is not too big


    while doc_index < len(search_hits):        
        search_hit = search_hits[doc_index]
        doc_index += 1
        unique_url = search_hit['url']

        if unique_url in loaded_urls:
            continue

        # doc_md = await html_to_markdown(unique_url, "#body-inner")
        doc_md = search_hit['content_markdown']
        doc_trimmed = doc_md[:cfg.MAX_SOURCE_LENGTH]
        if (docs_length + len(doc_trimmed)) > cfg.MAX_CONTEXT_LENGTH:
            doc_trimmed = doc_trimmed[:cfg.MAX_CONTEXT_LENGTH - docs_length - 20]

        if len(doc_trimmed) == 0:
            break

        loaded_doc = {
            'page_content': doc_trimmed,
            'metadata': {            
                'source': unique_url,                                
            }
        }    
        print(f'loaded converted html to md doc, length= {len(doc_trimmed)}, url= {unique_url}')
        # pp.pprint(loaded_doc)

        docs_length += len(doc_trimmed)
        loaded_docs.append(loaded_doc)
        loaded_urls.append(unique_url)
        loaded_search_hits.append(search_hit)        

        if docs_length >= cfg.MAX_CONTEXT_LENGTH:
            print(f'MAX_CONTEXT_LENGTH: {cfg.MAX_CONTEXT_LENGTH} exceeded, loaded {len(loaded_docs)} docs.')
            break

    not_loaded_urls = []
    for hit in search_hits:
        url = hit['url']
        if url not in loaded_urls and url not in not_loaded_urls:
            not_loaded_urls.append(url)

    durations['download_docs'] = timeit.default_timer() - start

    print(f'stuffed source document urls:')
    # pp.pprint(loaded_urls)

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
            'title': next((hit['lvl0'] for hit in search_hits if hit['url'] == context.source), None),
        }
        for context in result['function'].relevant_contexts]
        rag_success = result['function'].i_dont_know != True
    else:
        relevant_sources = []
        # rag_success = None

    response = {
        'result': result['function'].helpful_answer,        
        'input_language': extract_search_queries.userInputLanguage,
        'rag_success': rag_success,
        'search_queries': extract_search_queries.searchQueries,
        'source_urls': loaded_urls,
        'source_documents': loaded_docs,
        'relevant_urls': relevant_sources,
        'not_loaded_urls': not_loaded_urls,
        'durations': durations,
    }

    # pp.pprint(response)

    return response