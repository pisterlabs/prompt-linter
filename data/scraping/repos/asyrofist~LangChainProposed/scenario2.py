import os, json, argparse, torch
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import  FAISS #, ElasticVectorSearch, Pinecone, Weaviate
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI, HuggingFaceHub
from langchain.callbacks import get_openai_callback
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def main():

    parser = argparse.ArgumentParser(description='Final Project: scenario2')
    parser.add_argument('--config_path', type=str, default='config_scenario2.json', help='path to the config.json file')

    # variable parser explanation
    args = parser.parse_args()

    config_path = args.config_path
    config = load_config(config_path)

    # Extract parameters from the config dictionary
    data_dir_param = config['data_dir_param']
    save_param = config['save_param']
    repo_id_param = config['repo_id_param']
    huggingface_token_param = config['huggingface_token_param']
    openapi_token_param = config['openapi_token_param']
    openapi_model_param = config['openapi_model_param']
    chain_type_param = config['chain_type_param']
    device_param = config['device_param']
    temperature_param = config['temperature_param']
    max_length_param = config['max_length_param']
    pad_token_id_param = config['pad_token_id_param']
    top_p_param = config['top_p_param']
    repetation_penalty_param = config['repetation_penalty_param']
    chunk_size_param = config['chunk_size_param']
    chunk_overlap_param = config['chunk_overlap_param']
    huggingface_active = config['huggingface_active']
    openapi_active = config['openapi_active']
    queries = config['queries']

    # load data directory
    data_dir = data_dir_param
    data_list = os.listdir(data_dir)

    data_article= []
    for file_name in data_list:
        file_path = os.path.join(data_dir, file_name)
        with open(file_path, 'r', encoding='iso-8859-1') as f:
            article_data = f.read()
            data_article.append(article_data)

    # location of the pdf file/files.
    for num in data_list:
        reader = PdfReader(os.path.join(data_dir, num))
        # read data from the file and put them into a variable called raw_text
        raw_text = ''
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text

        # We need to split the text that we read into smaller chunks so that during information retreival we don't hit the token size limits.
        text_splitter = CharacterTextSplitter(
            separator = "\n",
            chunk_size = chunk_size_param,
            chunk_overlap  = chunk_overlap_param,
            length_function = len,
        )
        texts = text_splitter.split_text(raw_text)
        print(len(texts))

        # Download embeddings from OpenAI
        if huggingface_active == True:
            os.environ['HUGGINGFACEHUB_API_TOKEN'] = huggingface_token_param
            embeddings= HuggingFaceEmbeddings()

            llm_hf = HuggingFaceHub(
                      repo_id= repo_id_param,
                      model_kwargs= {'temperature': temperature_param,
                                      'max_length': max_length_param,
                                      'pad_token_id': pad_token_id_param,
                                      'top_p': top_p_param,
                                      'device': device_param,
                                      'repetition_penalty': repetation_penalty_param}
            )

            chain = load_qa_chain(llm= llm_hf, chain_type= chain_type_param)

        if openapi_active == True:
            os.environ["OPENAI_API_KEY"] = openapi_token_param
            embeddings = OpenAIEmbeddings()
            chain = load_qa_chain(llm= OpenAI(model= openapi_model_param), chain_type= chain_type_param)

        docsearch = FAISS.from_texts(texts, embeddings)
        responses = []

        for query in queries:
            docs = docsearch.similarity_search(query)
            #response = chain.run(input_documents=docs, question=query, parameters={'truncation': 'only_first'})
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
