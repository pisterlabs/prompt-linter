import box
import timeit
import yaml
import pprint
import tempfile
import os

from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import UnstructuredMarkdownLoader
from docs_qa.chains import build_llm
from docs_qa.extract_search_terms import run_query_async
from .html_to_markdown import html_to_markdown
from docs_qa.typesense_search import typesense_search_multiple


pp = pprint.PrettyPrinter(indent=2)

# Import config vars
with open('docs_qa/config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))



async def rag_with_typesense(user_input):
    extract_search_queries = await run_query_async(user_input)
    pp.pprint(extract_search_queries)

    search_response = await typesense_search_multiple(extract_search_queries[:10])
    pp.pprint(search_response)

    documents = [
        {
            'id': document['document']['id'],
            'url': document['document']['url_without_anchor']
        }
        for result in search_response['results']
        for hit in result['grouped_hits']
        for document in hit['hits']
    ]    

    # print(f'Document IDs')
    # pp.pprint(documents)

    unique_urls = list(set([document['url'] for document in documents]))
    print(f'Unique URLs')
    pp.pprint(unique_urls)

    download_start = timeit.default_timer()
    # download source HTML and convert to markdown - should be done by scraper    
    with tempfile.TemporaryDirectory() as temp_dir:
        md_docs = [
            {
                'page_content': await html_to_markdown(unique_url, "#body-inner"),
                'metadata': { 
                    'source': unique_url,
                    'file_path': os.path.join(temp_dir, unique_url.replace('/', '_').replace('https:', '') + '.md')
                },                
            }
            for unique_url in unique_urls
        ]
        print(f'html_to_md docs:')
        pp.pprint(md_docs)
        
        loaded_docs = []
        docs_length = 0


        for doc in md_docs:
            with open(doc['metadata']['file_path'], 'w') as f:
                f.write(doc['page_content'])
                f.flush()
                loaded_doc = UnstructuredMarkdownLoader(doc['metadata']['file_path']).load()[0]
                docs_length += len(loaded_doc.page_content)
                if docs_length < cfg.MAX_CONTEXT_LENGTH:
                    loaded_docs.append(loaded_doc)
                

        # print(f'loaded markdown docs')
        # pp.pprint(loaded_docs)

    download_end = timeit.default_timer()    
    print(f"Time to download and convert source URLs: {download_end - download_start} seconds")

    print(f'Starting load_qa_chain...')
    chain_start = timeit.default_timer()

    llm = build_llm()
    chain = load_qa_chain(llm, chain_type="stuff", verbose=False)
    result = chain.run(input_documents=loaded_docs, question=user_input)
    chain_end = timeit.default_timer()

    response = {
        'result': result,
        'source_documents': md_docs,
        'source_urls': unique_urls,
        'search_terms': extract_search_queries.searchQueries,
    }
    print(f"Time to run load_qa_chain: {chain_end - chain_start} seconds")

    pp.pprint(response)

    return response