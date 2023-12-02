import pandas as pd
from llama_index import Document
from llamaindex_utils import set_service_context, get_index

def gen_citation_db(service_context, index_name):
    '''
    Generate the vector db using the reasons for citation and save to disk.
    '''
    data = pd.read_csv('../data/FRB_citations_23.csv')

    # each doc should contain one reason for citation
    documents = []
    for i, row in data.iterrows():
        metadata = {
            'doc_id': str(i),
        }
        reasons = row['reasons'].replace("\n", "").split(';')
        for reason in reasons:
            if '*' in reason[:2]:
                # don't include ambiguous ones for now
                continue
            document = Document(text=reason, metadata=metadata)
            document.excluded_embed_metadata_keys = ['doc_id']
            documents.append(document)

    # create index
    Index = get_index(index_name)
    if Index is None:
        return
    index = Index.from_documents(
        documents,
        service_context=service_context,
        show_progress=True,
    )
    index.storage_context.persist(f'../data/llamaindex{index_name}_openaiEmbed_citation_db')

def gen_abstract_db(service_context, index_name):
    data = pd.read_csv('../data/FRB_abstracts.csv')

    # each doc should contain title and abstract
    documents = []
    for i, row in data.iterrows():
        metadata = {
            'doc_id': str(i),
        }
        text = row['title'] + ': ' + row['abstract']
        text = text.replace("\n", " ").lower()
        document = Document(text=text, metadata=metadata)
        document.excluded_embed_metadata_keys = ['doc_id']
        documents.append(document)

    # create index
    Index = get_index(index_name)
    if Index is None:
        return
    index = Index.from_documents(
        documents,
        service_context=service_context,
        show_progress=True,
    )
    index.storage_context.persist(f'../data/llamaindex{index_name}_openaiEmbed_abstract_db')

def main():
    service_context = set_service_context()
    for index_name in ['VectorStoreIndex',
                       'SimpleKeywordTableIndex',
                       'RAKEKeywordTableIndex']:
        gen_citation_db(service_context, index_name)
        gen_abstract_db(service_context, index_name)

if __name__ == '__main__':
    main()
