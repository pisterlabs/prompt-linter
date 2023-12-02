import os
import time
import sagemaker
import boto3
import openai
from botocore.exceptions import ClientError
from PyPDF2 import PdfReader
from textractor import Textractor
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.docstore import InMemoryDocstore
from langchain.agents import load_tools, initialize_agent
from langchain.llms import OpenAI
from langchain import VectorDBQA
from langchain.chains import RetrievalQA
from langchain import FAISS
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from chain import get_new_chain1
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
import faiss
# BabyAGI
from babyAGI import Optional, BabyAGI
import sagemakerLLM
from bloom import bloom_inference

# API keys and tokens
openai.api_key = os.environ.get('OPENAI_API_TOKEN')
serp_api_token = os.environ.get('SERP_API_TOKEN')
wolframalpha_api_token = os.environ.get('WOLFRAMALPHA_API_TOKEN')
stabilityai_api_token = os.environ.get('STABILITYAI_API_TOKEN')

# Set up Amazon textract
textract_client = boto3.client('textract')
output_file = '/tmp/textract_pdf_2_text.txt'

# Create a SageMaker session
sagemaker_session = sagemaker.Session()

# Get the default S3 bucket name
default_bucket_name = sagemaker_session.default_bucket()
s3_client = boto3.client('s3')

# Default embedding model and llm model
embedding_model = OpenAIEmbeddings(openai_api_key=openai.api_key)
llm_model = OpenAI(temperature=0, openai_api_key=openai.api_key)

# Initialize default vectorstore as empty
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
faiss_store = FAISS(embedding_model.embed_query, index, InMemoryDocstore({}), {})

def parse_credentials(file_path):
    credentials = {}
    with open(file_path, 'r') as file:
        current_user = None
        for line in file:
            line = line.strip()
            if line.startswith('[') and line.endswith(']'):
                current_user = line[1:-1]
                credentials[current_user] = {}
            elif '=' in line and current_user is not None:
                key, value = line.split('=', 1)
                credentials[current_user][key] = value
    return credentials

def get_key_from_credential_file(user, key_name, credential_file_path):
    credentials = parse_credentials(credential_file_path)

    if user in credentials:
        user_credentials = credentials[user]
        if key_name in user_credentials:
            return user_credentials[key_name]
        else:
            raise KeyError(f"'{key_name}' not found for user '{user}'.")
    else:
        raise KeyError(f"User '{user}' not found in the credential file.")
        
# Function for openai
def openai_create(prompt, model_choice):
    if len(prompt)>=4096:
      prompt=prompt[0:4095]
    response = openai.ChatCompletion.create(
            #model="gpt-3.5-turbo-0301",
            model=model_choice,
            messages=[
                {"role": "system", "content": "You are a helpful assistant in answering question, completing sentences and rewrite documents."},
                {"role": "user", "content": prompt}
                ],
    )
    return response['choices'][0]['message']['content']


def transcribe(audio, state=""):
    #time.sleep(0.5)
    text = p(audio)["text"]
    #state += text + " "
    state = "HF Pipeline ASR done! "
    return text, state


def convert_to_16_bit_wav(data):
    # Based on: https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.write.html
    #breakpoint()
    if data.dtype == np.float32:
        warnings.warn(
            "Audio data is not in 16-bit integer format."
            "Trying to convert to 16-bit int format."
        )
        data = data / np.abs(data).max()
        data = data * 32767
        data = data.astype(np.int16)
    elif data.dtype == np.int32:
        warnings.warn(
            "Audio data is not in 16-bit integer format."
            "Trying to convert to 16-bit int format."
        )
        data = data / 65538
        data = data.astype(np.int16)
    elif data.dtype == np.int16:
        pass
    elif data.dtype == np.uint8:
        warnings.warn(
            "Audio data is not in 16-bit integer format."
            "Trying to convert to 16-bit int format."
        )
        data = data * 257 - 32768
        data = data.astype(np.int16)
    else:
        raise ValueError("Audio data cannot be converted to " "16-bit int format.")
    return data

def openai_asr(audio_file, state, model_choice):
    if not (audio_file is None):
      if os.path.isfile(audio_file):
        audio = open(audio_file, "rb")
        transcribe = openai.Audio.transcribe("whisper-1", audio)
        text = transcribe['text']
        if model_choice == 'dgidp':
            out, state2 = langchain_idp(text, state, model_choice)
        else:
            out, state2 = chatgpt_clone(text, state, model_choice)
        return out, state2
      else:
        return "Wrong audio format", state
    
    
def clear_callback(interface):
    interface.inputs[0].reset()
    
# Display an image
def show_image(input_file):
    input_image = Image.open(input_file.name)
    return input_image

# Lang Chain search
def langchain_search(input_str, history):
    if input_str != "":
        langchain_llm = OpenAI(temperature=0, model='text-davinci-003', openai_api_key=openai.api_key)
        langchain_memory = ConversationBufferMemory(memory_key="chat_history")
        history = history or []
        serapi_search = SerpAPIWrapper(serpapi_api_key=serp_api_token)
        wolfram_chain = LLMMathChain(llm=langchain_llm, verbose=True)

        tools = [
            Tool(
                name = "Serapi",
                func=serapi_search.run,
                description="useful for when you need to answer questions about current events via SerpAPI"
            ),
            Tool(
                name="Calculator",
                func=wolfram_chain.run,
                description="useful for when you need to answer questions about math"
            )
        ]
        agent = initialize_agent(tools, langchain_llm, agent="conversational-react-description", memory=langchain_memory, verbose=True)
        output = agent.run(input_str)
        history.append((input_str, output))
        return history, history

# Amazon textract extract text from pdf files
def pdf_2_text(input_pdf_file, history):
    #output_file = '/tmp/textract_pdf_2_text.txt'
    history = history or []
    key = 'input-pdf-files/{}'.format(os.path.basename(input_pdf_file.name))
    try:
        response = s3_client.upload_file(input_pdf_file.name, default_bucket_name, key)
        #print("File uploaded to S3 successfully")
    except ClientError as e:
        print("Error uploading file to S3:", e)
    
    s3_object = {'Bucket': default_bucket_name, 'Name': key}
    response = textract_client.start_document_analysis(
        DocumentLocation={'S3Object': s3_object},
        FeatureTypes=['TABLES', 'FORMS']
    )
    job_id = response['JobId']
    #print("Started Textract job with ID:", job_id)
    
    while True:
        response = textract_client.get_document_analysis(JobId=job_id)
        status = response['JobStatus']
        #print("Job status:", status)
        if status in ['SUCCEEDED', 'FAILED']:
            break
        time.sleep(5)
    
    if status == 'SUCCEEDED':
        with open(output_file, 'w') as output_file_io:
            for block in response['Blocks']:
                if block['BlockType'] in ['LINE', 'WORD']:
                    output_file_io.write(block['Text'] + '\n')
        with open(output_file, "r") as file:
            first_512_chars = file.read(512).replace("\n", "").replace("\r", "").replace("[", "").replace("]", "") + " [...]"
        history.append(("Document conversion", first_512_chars))
        #history = history.append(("", first_512_chars))
        return history, history

def get_faiss_store():
    with open("docs.pkl", 'rb') as f:
        faiss_store = pickle.load(f)
        return faiss_store
    
def langchain_idp(query_input, history, model_choice):
    separator = '\n'
    overlap_count = 100
    chunk_size = 1000
    history = history or []
    #if len(texts) > 0 :
    loader = TextLoader(output_file)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(separator=separator, chunk_overlap=overlap_count, chunk_size=chunk_size, length_function=len)
    texts = text_splitter.split_documents(documents)
    docsearch = Chroma.from_documents(texts, embedding_model)
    
    if model_choice=="j2-jumbo-instruct":
        llm = sagemakerLLM.SageMakerLLM()
        history.append((query_input, llm(query_input)))
        
    if  model_choice=="flan-ul2":
        embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
        vectorstore = get_faiss_store()
        flan_ul = HuggingFaceHub(repo_id="google/flan-ul2",
                                model_kwargs={"temperature":0.1, "max_new_tokens":200},
                                huggingfacehub_api_token=hf_api_token)
        qa_chain = get_new_chain1(vectorstore, flan_ul, flan_ul, isFlan=True)
        response = qa_chain.run(query_input)
    elif model_choice=="flan-t5-xl":
        template = """Question: {question}
        Answer: Let's think step by step."""
        prompt = PromptTemplate(template=template, input_variables=["question"])
        qa_chain = LLMChain(prompt=prompt, llm=HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature":0, "max_length":256}, huggingfacehub_api_token=hf_api_token))
        #response = qa_chain.run(query_input)
        history.append((query_input, qa_chain.run(query_input)))
    elif model_choice=="bloom":
        max_length = 128
        sample_or_greedy = 'Greedy'
        history.append((query_input,  bloom_inference(query_input,  max_length, sample_or_greedy, seed=42)))
    # elif model_choice=="bedrock":
    #     history.append((query_input, bedrock(query_input)))
    elif model_choice=="dgidp":
        llm = sagemakerLLM.SageMakerLLM()
        #llm = OpenAI(model_name='text-davinci-003', temperature=0, openai_api_key=openai.api_key)
        #llm = OpenAI(model_name='gpt-3.5-turbo', temperature=0, openai_api_key=openai.api_key)
        qa_chain = VectorDBQA.from_chain_type(llm=llm, chain_type='stuff', vectorstore=docsearch)
        #qa_chain = RetrievalQA.from_llm(llm=llm, vectorstore=vectordb)
        response = qa_chain({'query': query_input}, return_only_outputs=True)
        history.append((query_input, response['result']))
    elif model_choice=="babyagi":
        # Logging of LLMChains
        verbose = False
        # If None, will keep on going forever
        max_iterations: Optional[int] = 3
        baby_agi = BabyAGI.from_llm(
            llm=llm_model, vectorstore=faiss_store, verbose=verbose, max_iterations=max_iterations
        )
        baby_agi({"objective": query_input})
        # Process results
        index = list(faiss_store.index_to_docstore_id)[-1]
        response = faiss_store.docstore.search(faiss_store.index_to_docstore_id[index]).page_content
        history.append((query_input, response))

    elif model_choice=="gpt-3.5":
        model_name = 'gpt-3.5-turbo'
        history.append((query_input, openai_create(query_input, model_name)))
    else:
        history.append((query_input, "Not implemented"))
    #response = qa_chain({'query': query_input}, return_only_outputs=True)
    #history.append((query_input, response['result']))
    return history, history
        
