import json
import argparse
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

def extract_good_examples_from_jsonl(jsonl_path):
    """Extract examples labeled "Good" from a JSONL dataset exported from Label Studio. 

    Args:
        jsonl_path (str): Path to the JSONL dataset. Note: Json files from label studio are really JSONL. 

    Returns:
        list: List of examples as [{'prompt':'', 'response':'',...}]
    """
    with open(jsonl_path, 'r') as f:
        data = json.load(f)
    
    return [
        record['data'] for record in data 
        if any(
            result['value']['choices'][0] == "Good" 
            for annotation in record['annotations'] 
            for result in annotation['result']
        )
    ]

def main(json_path, persist_directory):
    """Main function to extract and store good examples in vectordb."""
    # Initialize the vector store with a given directory and embedding function
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=OpenAIEmbeddings())

    # Extract "good" examples and convert them to Documents
    good_examples = extract_good_examples_from_jsonl(json_path)
    docs = [Document(page_content=str(example), metadata={"source": json_path}) for example in good_examples]

    # Add documents to the vector store and persist
    vectorstore.add_documents(docs)
    vectorstore.persist()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 'good' labeled examples from a JSONL file and add to vectordb.")
    parser.add_argument("--json_path", required=True, help="Path to the JSONL file with labeled examples.")
    parser.add_argument("--persist_dir", required=True, help="Directory where vectordb embeddings are stored.")
    args = parser.parse_args()
    
    main(args.json_path, args.persist_dir)
