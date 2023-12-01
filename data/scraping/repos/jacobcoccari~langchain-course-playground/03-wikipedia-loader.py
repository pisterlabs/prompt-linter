from langchain.document_loaders import WikipediaLoader

docs = WikipediaLoader(
    query="George H. W. Bush vomiting incident",
    load_max_docs=2,
).load()
len(docs)

print(docs[0].metadata)  # meta-information of the Document

print(docs[0].page_content)  # a content of the Document
