from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def main():

    # set string parameters
    parser = argparse.ArgumentParser(description='Final Project: Scenario1')
    parser.add_argument('--config_path', type=str, default='config_scenario1.json', help='path to the JSON config file')

    # variable parser explanation
    args = parser.parse_args()

    config_path = args.config_path
    config = load_config(config_path)

    pdf_param = config['pdf_param']
    huggingface_token_param = config['huggingface_token_param']
    openapi_token_param = config['openapi_token_param']
    save_param = config['save_param']
    chunk_size_param = config['chunk_size_param']
    chunk_overlap_param = config['chunk_overlap_param']
    queries = config['queries']

    # location of the pdf file/files
    doc_reader = PdfReader(pdf_param)

    # read data from the file and put them into a variable called raw_text
    raw_text = ''
    for i, page in enumerate(doc_reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

    print(raw_text[:100])

    # Splitting up the text into smaller chunks for indexing
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size= chunk_size_param,
        chunk_overlap= chunk_overlap_param,  # striding over the text
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    # Download embeddings from OpenAI
    if huggingface_active == True:
        os.environ['HUGGINGFACEHUB_API_TOKEN'] = huggingface_token_param
        embeddings= HuggingFaceEmbeddings()

    if openai_active == True:
        os.environ["OPENAI_API_KEY"] = openapi_token_param
        embeddings = OpenAIEmbeddings()

    docsearch = embeddings.from_texts(texts) # Indexing process without FAISS
    responses = []

    for query in queries:
        docs = docsearch.similarity_search(query)
        response = chain.run(input_documents=docs, question=query)
        responses.append(response)
        print(response)

    # Sample string
    data = {
      "author": response[0],
      "title": response[1],
      "Theoretical/ Conceptual Framework": response[2],
      "Research Question(s)/ Hypotheses": response[3],
      "methodology": response[4],
      "Analysis & Results study": response[5],
      "conclusion": response[6],
      "Implications for Future research": response[7],
      "Implication for practice": response[8],
    }

    # Save the JSON object to a file
    with open(save_param, 'w') as f:
        json.dump(data, f)

if __name__ == "__main__":
   main()
