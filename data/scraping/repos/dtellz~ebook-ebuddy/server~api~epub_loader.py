from langchain.document_loaders import UnstructuredEPubLoader

def loadEpub(file):
    print('Loading EPUB file...')
    epub = UnstructuredEPubLoader(file)
    print('EPUB file loaded.')
    documents = epub.load()
    print('EPUB file documented.')
    # return documents


