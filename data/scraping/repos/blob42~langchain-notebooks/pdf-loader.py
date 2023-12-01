r"""°°°
## Loading PDF
°°°"""
#|%%--%%| <ut22SE2PmJ|EQL3ZDG6Dt>

from langchain.document_loaders import PagedPDFSplitter

loader = PagedPDFSplitter("./documents/layout-parser-paper.pdf")
pages = loader.load_and_split()

#|%%--%%| <EQL3ZDG6Dt|6LWg1c7vN6>
r"""°°°
Documents can be retrived with page numbers
°°°"""
#|%%--%%| <6LWg1c7vN6|0kFnbEI7yL>

from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

#|%%--%%| <0kFnbEI7yL|KkXwCS4JHN>

faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings() )

# Find docs (ie pages) most similar to query
# k: number of docs similar to query
docs = faiss_index.similarity_search("How will the community be engaged ?", k=2)

#|%%--%%| <KkXwCS4JHN|RDajVoEdqh>
# get page numbers + content,  similar to query 
for doc in docs:
    print("\n----\n")
    print("page: " + str(doc.metadata["page"] + 1))
    print("content:")
    print(str(doc.page_content))

#|%%--%%| <RDajVoEdqh|cqoPocvVBS>

