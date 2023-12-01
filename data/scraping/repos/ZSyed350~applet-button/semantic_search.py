import os
import cohere

def load_files_from_directory(directory):
    """Load all code files from a directory into a list."""
    documents = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):  # assuming Python files, adjust the extension if needed
                with open(os.path.join(root, file), 'r') as f:
                    content = f.read()
                    documents.append(content)
    return documents

def rerank_files_based_on_query(directory, query):
    """Rerank code files in a directory based on a query."""
    
    # Initialize the Cohere client
    co = cohere.Client("{apiKey}")
    
    # Load the code files from the directory
    documents = load_files_from_directory(directory)
    
    # Use Cohere's rerank API
    results = co.rerank(query=query, documents=documents, top_n=3, model="rerank-multilingual-v2.0")
    
    # Returning the top ranked documents
    return [documents[i] for i in results.ranks]

# Example usage:
directory = '/applets'
query = 'code to create a Flask app'
ranked_files = rerank_files_based_on_query(directory, query)

# Display or handle the results as needed
for file in ranked_files:
    print(file)
    print("="*50)