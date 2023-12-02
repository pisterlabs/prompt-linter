# Directory loader for text files

from langchain.document_loaders import DirectoryLoader, TextLoader

def load_text(directory):  # Load and process the text files
    loader = DirectoryLoader(directory, glob = "./*.txt", loader_cls = TextLoader, show_progress = True)    
    texts = loader.load()
    print(f"Loaded {len(texts)} text files!")
    return texts