from dotenv import load_dotenv,find_dotenv
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, DirectoryLoader, PyPDFLoader
from langchain.schema import prompt_template
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline
from langchain.chains import SimpleSequentialChain, SequentialChain
from instruct_pipeline import InstructionTextGenerationPipeline
from langchain import PromptTemplate
import transformers
import torch
import pinecone
import os
import sys

def main():
    
    name = 'mosaicml/mpt-30b'

    config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)
    #config.max_seq_len = 8192
    #config.attn_config['attn_impl'] = 'triton'  # change this to use triton-based FlashAttention
    config.init_device = 'cuda:0'  # For fast initialization directly on GPU!

    load_8bit = True
    tokenizer = AutoTokenizer.from_pretrained(name)  # , padding_side="left")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        name,
        config=config,
        torch_dtype=torch.bfloat16,  # Load model weights in bfloat16
        trust_remote_code=True,
        load_in_8bit=load_8bit,
        device_map="auto",
    )

    model.eval()
    if torch.__version__ >= "2":
        model = torch.compile(model)

    print("--PIPELINE INIT--")
    pipe = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)

    llm = HuggingFacePipeline(pipeline = pipe, model_kwargs = {'temperature':0})

    pdf_path = sys.argv[1]
    loader = PyPDFLoader(pdf_path)

    ## Other options for loaders
    # loader = UnstructuredPDFLoader("../data/field-guide-to-data-science.pdf")
    ### This one is used to load an online pdf:
    # loader = OnlinePDFLoader("https://wolfpaulus.com/wp-content/uploads/2017/05/field-guide-to-data-science.pdf")

    data = loader.load()

    # Note: If you're using PyPDFLoader then it will split by page for you already
    print(f'You have {len(data)} document(s) in your data')
    # print (f'There are {len(data[0].page_content)} characters in your document')

    """
    directory = '/content/data'
    documents = load_docs(directory)
    len(documents)
    """

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    texts = text_splitter.split_documents(data)
    print(f'Now you have {len(texts)} documents')

    print(texts)

    # Check to see if there is an environment variable with your API keys, if not, use what you put below
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', '9fa8ba9d-344d-4466-8e7e-78f825ad7caf')
    PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', 'gcp-starter')  # You may need to switch with your env

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    # initialize pinecone
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_API_ENV  # next to api key in console
    )

    index_name = "example"  # put in the name of your pinecone index here
    docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)

    query = "What category does this insurance claim belong to? The options are 'Polizze', 'Sinistri' and 'Area Commerciale'"  
    category_related_docs = docsearch.similarity_search(query)
    
    context = """You are a text classifier of insurance claims.
    Given the following extracted parts of a long document and a question, classify and label
    the document with one of these three classes: Polizze, Sinistri and Area Commerciale.
    Here is a detailed description of the meaning of each class:
    The 'Polizze' class groups all the insurance claims sent by customers to report issues related to the contract established between the customer and the referring insurance company are included.
    This includes any requested insurance type: life insurance, auto insurance, health insurance, and more.
    The 'Sinistri' class encompasses all cases where customers contest issues related to incidents, which may include accidental damages and road accidents.
    he 'Area Commerciale' class groups all the insurance claims are reviewed where customers contest the lack of assistance from the insurance company despite repeated requests.
    Additionally, this section includes customer inquiries, such as requests for information about insurance types, quotation requests, issues related to the website and application
    """

    macrocategory_prompt_template = """Use the following pieces of context to answer to the provided query. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}

    Question: {question}
    Answer in Italian:""" 

    macrocategory_prompt = PromptTemplate(
        template=macrocategory_prompt_template,
        input_variables=["context","question"]
    )
    
    chain = load_qa_chain(llm=llm, prompt=macrocategory_prompt, output_key="macrocategory_result", chain_type="stuff")
    
    # Run the macrocategory chain to classify the macrocategory
    macrocategory_result = chain({"input_documents": category_related_docs, "question": query})
    
    print(f"Predicted macrocategory: {macrocategory_result}")
    
    ###############################################################################################################
    
    query = "What branch does this insurance claim belong to? The options are 'Auto', 'Vita' and 'Altri rami'."
    
    context = """
        You are again a text classifier. Given your previous answer containing the class assigned to the insurance claim,
        classify the insurance claim for the Polizze category into three possible branches: 'Auto', 'Vita' and 'Altri rami'.
        'Auto' includes all the insurance claims related to car insurance.
        'Vita' includes all the insurance claims related to life insurance.
        'Altri rami' includes all the insurance claims that do not belong to the 'Auto' or 'Vita' branches, for example
        health insurance, home insurance, theft insurance, fire insurance, technological risks and more.
        """

    question_branch = """What branch does this insurance claim belong to? The options are 'Auto', 'Vita' and 'Altri rami'.
    The 'Altri rami' branch includes all the insurance claims that do not belong to the 'Auto' or 'Vita' branches."""
        
    branch_related_docs = docsearch.similarity_search(question_branch)
    
    branch_prompt_template = """Use the following pieces of context to answer to the provided query. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Macrocategory: {macrocategory_result}
    Context: {context}

    Question: {question}
    Answer in Italian:""" 

    branch_prompt = PromptTemplate(
        template=branch_prompt_template,
        input_variables=["macrocategory_result","context","question"]
    )

    chain_two = load_qa_chain(pipeline=pipeline, prompt=branch_prompt, output_key="branch_result")

    # Run the second chain to classify the branch
    branch_result = chain_two({"input_documents": branch_related_docs, "question": query}, return_only_outputs=True)

    print(f"Branch result: {branch_result}")
    
    # Define a sequential chain using the two chains above: the second chain takes the output of the first chain as input
    #overall_chain = SimpleSequentialChain(chains=[chain, chain_two], verbose=True)

    
    overall_chain = SequentialChain(
        chains=[chain, chain_two],
        input_variables=["context", "question"],
        # Here we return multiple variables
        output_variables=["macrocategory_result", "branch_result"],
        verbose=True)
    

    # Run the overall chain
    overall_result = overall_chain({"input_documents": data}, return_only_outputs=True)
    print(f"Overall result: {overall_result}")

"""
def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents
"""

if __name__ == "__main__":
    main()