#from transformers import pipeline
import gradio as gr
import time
import os
import openai
from textractor import Textractor
import magic
from PyPDF2 import PdfReader
import requests
import boto3
import json
import pickle
import sagemaker
# Adding LangChain
from langchain.agents import load_tools, initialize_agent
from langchain.llms import OpenAI
from langchain import VectorDBQA
from langchain.chains import RetrievalQA
from langchain import FAISS
from langchain.chains.conversation.memory import ConversationBufferMemory
# Question Answering over Docs
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from botocore.exceptions import ClientError
from langchain import PromptTemplate, HuggingFaceHub, LLMChain

# Local chain property
from chain import get_new_chain1

# BabyAGI
from babyagi import Optional, BabyAGI
#from langchain import OpenAI
#from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
#from langchain.embeddings import OpenAIEmbeddings
import faiss


# Create a SageMaker session
sagemaker_session = sagemaker.Session()
textract_client = boto3.client('textract')
# Get the default S3 bucket name
default_bucket_name = sagemaker_session.default_bucket()

#access_token = os.environ.get('hf_api_token')
openai.api_key = os.environ.get('openai_api_token')
hf_api_token = os.environ.get('hf_api_token')
serp_api_token = os.environ.get('serp_api_token')
wolframalpha_api_token = os.environ.get('wolframalpha_api_token')
stabilityai_api_token = os.environ.get('stabilityai_api')


#p = pipeline("automatic-speech-recognition", use_auth_token=access_token)
API_URL = "https://api-inference.huggingface.co/models/bigscience/bloom"
headers = {"Authorization": "Bearer " + hf_api_token}
extractor = Textractor(profile_name="default")
start_sequence = "\nAI:"
restart_sequence = "\nHuman: "
prompt = "How can I help you today?"
s3_client = boto3.client('s3')

# Lang Chain
langchain_llm = OpenAI(temperature=0, model='text-davinci-003', openai_api_key=openai.api_key)
langchain_memory = ConversationBufferMemory(memory_key="chat_history")
#langchain_agent = initialize_agent(tools, langchain_llm, agent="conversational-react-description", memory=memory, verbose=True)
output_file = '/tmp/textract_pdf_2_text.txt'


# Define function needed for Bedrock
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

aws_access_key_id = get_key_from_credential_file('qa', 'aws_access_key_id', '/home/alfred/.aws/credentials')
aws_secret_access_key = get_key_from_credential_file('qa', 'aws_secret_access_key', '/home/alfred/.aws/credentials')
bedrock_model_arn = "arn:aws:amazon-bedrock::aws:built-in-model/bedrock-large-01"
bedrock_url = 'https://amazo-loadb-10wvy7j77n07w-1803419470.us-east-1.elb.amazonaws.com/'
MODEL_IDENTIFIER = "AMAZON.bedrock-text-similarity-embedding-01"
bedrock_client = boto3.client(
    service_name='sagemakerbedrock',
    region_name='us-east-1',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    endpoint_url=bedrock_url
)

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

def ocr_aws(tempfile, history):
    # Get the filename from tempfile type
    if os.path.isfile(tempfile.name):
      document = extractor.detect_document_text(file_source=tempfile.name)
      history = history or []
      my_string = str(document.lines).replace("\n", "").replace("\r", "").replace("[", "").replace("]", "")
      history.append(("", my_string))
      return history, history

def ocr_opensource(tempfile, history):
    # Get the filename from tempfile type
    my_string = ""
    if os.path.isfile(tempfile.name):
      if magic.from_file(tempfile.name, mime=True) == "application/pdf":
        reader = PdfReader(tempfile.name)
        for i in range(len(reader.pages)):
          my_string += reader.pages[i].extract_text().replace("\n", " ").replace("\r", " ")
    else:
      document = extractor.detect_document_text(file_source=tempfile.name)
      my_string = str(document.lines).replace("\n", "").replace("\r", "").replace("[", "").replace("]", "")
    history = history or []
    history.append(("", my_string))
    return history, history


'''
gr.Interface(
    #fn=transcribe,
    fn=openai_asr,
    inputs=[
        gr.Audio(source="microphone", type="filepath", show_label=True),
        "state"
    ],
    outputs=[
        "textbox",
        "state"
    ],
    live=True).launch(debug=True,server_name="0.0.0.0", share=False, height=768)
'''

# Bedrock
def truncate_list(input_list, stop_token):
    truncated_list = []
    for token in input_list:
        if token == stop_token:
            break
        else:
            truncated_list.append(token)
    return truncated_list

def bedrock(input):
    MODEL_IDENTIFIER = 'AMAZON.bedrock-large-01'
    textGenerationConfig = {
        "maxTokenCount": 658,
        "minTokenCount": 0,
        "temperature": 2,
        "topP": 0.5,
        "beamCount": 2,
        "topK": 50,
        "repetitionPenalty": 1,
        "lengthPenalty": 1,
        "noRepeatNgramSize": 0,
        #"stopSequences": [],
    }
    query_response = bedrock_client.generate_text(inputText=input, modelIdentifier=MODEL_IDENTIFIER, textGenerationConfig=textGenerationConfig)
    return query_response['results'][0]['outputText']

# BLOOM
def query_hf_api(payload):
  response = requests.request("POST", API_URL, json=payload, headers={"Authorization": f"Bearer {hf_api_token}"})
  return response.json()
  #return json.load(response.content.decode("utf-8"))

def truncate_string(input_string, stop_token):
    tokens = input_string.split() # Split the input string into tokens
    output_tokens = [] # Create an empty list to store the output tokens

    # Loop through the tokens and add them to the output list until the stop token is encountered
    for token in tokens:
        if token == stop_token:
            break
        else:
            output_tokens.append(token)

    output_string = " ".join(output_tokens) # Join the output tokens into a string

    return output_string

def bloom_inference(input_sentence, max_length, sample_or_greedy, seed=42):
    if sample_or_greedy == "Sample":
        json_ = {
          "inputs": ''.join(input_sentence),
          "parameters": {
            "max_new_tokens": max_length,
            "top_p": 0.9,
            "do_sample": True,
            "seed": seed,
            "early_stopping": False,
            "length_penalty": 0.0,
            "eos_token_id": None,
          },
          "options":
          {
            "use_cache": False,
            "wait_for_model":True
          },
        }
    else:
        json_ = {
          "inputs": ''.join(input_sentence),
          "parameters": {
            "max_new_tokens": max_length,
            "do_sample": False,
            "seed": seed,
            "early_stopping": False,
            "length_penalty": 0.0,
            "eos_token_id": None,
          },
          "options":
          {
            "use_cache": False,
            "wait_for_model":True
          },
        }

    #payload = {"inputs": input_sentence, "parameters": parameters,"options" : {"use_cache": False} }
    data = query_hf_api(json_)

    if "error" in data:
        return (None, None, f"<span style='color:red'>ERROR: {data['error']} </span>")

    generation = data[0]["generated_text"].split(input_sentence, 1)[1]
    str_1 = generation.replace("\n", "").replace('\"','')
    return truncate_string(str_1, '^[\\s\\|]+$')
    '''
    return (
        before_prompt
        + input_sentence
        + prompt_to_generation
        + generation
        + after_generation,
        data[0]["generated_text"],
        "",
    )
    '''

# GPT-3.5
def chatgpt_clone(input, history, model_choice):
    if input != "":
    #  return "", history
    #else:
      history = history or []
      s = list(sum(history, ()))
      s.append(input)
      inp = ' '.join(s)
      if model_choice=="gpt-3.5":
        #model_name = 'gpt-3.5-turbo-0301'
        model_name = 'gpt-3.5-turbo'
        output = openai_create(inp, model_name)
      elif model_choice=="bedrock":
          output = bedrock(inp)
      elif model_choice=="bloom":
          #output = bloom(inp)
          max_length = 128
          sample_or_greedy = 'Greedy'
          output = bloom_inference(inp,  max_length, sample_or_greedy, seed=42)
      elif model_choice=="gpt-4":
          model_name = 'gpt-4'
          output = gpt4(inp, model_name)
      #output = openai_create(inp, model_choice)
      history.append((input, output))
      return history, history

def clear_callback(interface):
    interface.inputs[0].reset()
    
# Display an image
def show_image(input_file):
    input_image = Image.open(input_file.name)
    return input_image

# To be deleted
def chatgpt_clone_old(input, history):
    if input != "":
    #  return "", history
    #else:
      history = history or []
      s = list(sum(history, ()))
      s.append(input)
      inp = ' '.join(s)
      output = openai_create(inp)
      history.append((input, output))
      return history, history

# Lang Chain search
def langchain_search(input_str, history):
    if input_str != "":
        history = history or []
        serapi_search = SerpAPIWrapper(serpapi_api_key=serp_api_token)
        wolfram_chain = LLMMathChain(llm=langchain_llm, verbose=True)
        #serpapi_tool = load_tools(["serpapi", "llm-math"], llm=llm, serpapi_api_key=serp_api_token)
        #wolfram_tool = load_tools(['wolfram-alpha'], wolfram_alpha_appid=wolframalpha_api_token1)
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
    
# Lang Chain IDP
'''
def langchain_idp(input_str, prompt, history):
    if input_str != "":
        history = history or []
        loader = TextLoader(input_str)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_overlap=0, chunk_size=1000)
        texts = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
        docsearch = Chroma.from_documents(texts, embeddings)
        llm = OpenAI(model_name='text-davinci-003', temperature=0, openai_api_key=openai.api_key)
        qa_chain = VectorDBQA.from_chain_type(llm=llm, chain_type='stuff', vectorstore=docsearch)
        output = qa_chain({'query': prompt}, return_only_outputs=True)
        history.append((input_str, output['result']))
        return history, history
'''

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
    '''
    if model_choice=="gpt-3.5":
        embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
        docsearch = Chroma.from_documents(texts, embeddings)
        llm = OpenAI(model_name='text-davinci-003', temperature=0, openai_api_key=openai.api_key)
        qa_chain = VectorDBQA.from_chain_type(llm=llm, chain_type='stuff', vectorstore=docsearch)
    '''
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
    elif model_choice=="bedrock":
        history.append((query_input, bedrock(query_input)))
    elif model_choice=="dgidp":
        embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
        docsearch = Chroma.from_documents(texts, embeddings)
        #vectordb = FAISS.from_texts(texts, embeddings)
        llm = OpenAI(model_name='text-davinci-003', temperature=0, openai_api_key=openai.api_key)
        # :-( llm = OpenAI(model_name='gpt-3.5-turbo', temperature=0, openai_api_key=openai.api_key)
        qa_chain = VectorDBQA.from_chain_type(llm=llm, chain_type='stuff', vectorstore=docsearch)
        #qa_chain = RetrievalQA.from_llm(llm=llm, vectorstore=vectordb)
        response = qa_chain({'query': query_input}, return_only_outputs=True)
        history.append((query_input, response['result']))
    elif model_choice=="babyagi":
        # Define your embedding model
        embeddings_model = OpenAIEmbeddings(openai_api_key=openai.api_key)
        # Initialize the vectorstore as empty
        embedding_size = 1536
        index = faiss.IndexFlatL2(embedding_size)
        vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
        llm = OpenAI(temperature=0, openai_api_key=openai.api_key)
        # Logging of LLMChains
        verbose = False
        # If None, will keep on going forever
        max_iterations: Optional[int] = 1
        baby_agi = BabyAGI.from_llm(
            llm=llm, vectorstore=vectorstore, verbose=verbose, max_iterations=max_iterations
        )
        response = baby_agi({"objective": query_input})
        history.append((query_input, response))

    elif model_choice=="gpt-3.5":
        model_name = 'gpt-3.5-turbo'
        history.append((query_input, openai_create(query_input, model_name)))
    else:
        history.append((query_input, "Not implemented"))
    #response = qa_chain({'query': query_input}, return_only_outputs=True)
    #history.append((query_input, response['result']))
    return history, history
        

    
# Gradio portion
block = gr.Blocks()
last_message = prompt

with block:
    gr.HTML(
        f"""
          <div class="main-div">
            <div>
               <header>
               <h2>Dialogue Guided Intelligent Document Processing</h2>
               </header>
               <p>Dialogue Guided Intelligent Document Processing (DGIDP) is an innovative approach to extracting and processing information from documents by leveraging natural language understanding and conversational AI. This technique allows users to interact with the IDP system using human-like conversations, asking questions, and receiving relevant information in real-time. The system is designed to understand context, process unstructured data, and respond to user queries effectively and efficiently.</p> <p>While the text or voice chat accepts all major languages, the document upload feature only accepts files in English, German, French, Spanish, Italian, and Portuguese. The demo supports <u>multilingual text and voice</u> input, as well as <u>multi-page</u> documents in PDF, PNG, JPG, or TIFF format.</p>
            </div>
            <a href="https://www.buymeacoffee.com/alfredcs" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" height="32px" width="108px" alt="Buy Me A Coffee"></a>
            <br>
          </div>
        """
    )
    model_choice = gr.Dropdown(choices=["dgidp", "gpt-3.5", "babyagi", "bedrock", "bloom", "llama", "flan-t5-xl", "gpt4all", "gpt-4 (coming soon)"], label="Model selection", value="gpt-3.5")
    gr.HTML(f"""<hr style="color:blue>""")
    #file1 = gr.File(file_count="single")
    #upload = gr.Button("OCR")
    gr.HTML(f"""<hr style="color:blue>""")
    chatbot = gr.Chatbot().style(height=1750)
    #message = gr.Textbox(placeholder=prompt, lines=1)
    #audio = gr.Audio(source="microphone", type="filepath", show_label=True,height=550)
    #file1 = gr.File(file_count="single")
    state = gr.State()
    with gr.Row().style(equal_height=True):
      with gr.Column():
        message = gr.Textbox(placeholder=prompt, show_label=True)
        #textChat = gr.Button("Text Chat")
      with gr.Column():
        audio = gr.Audio(source="microphone", type="filepath", show_label=True)
        #voiceChat = gr.Button("Voice Chat")
    with gr.Row().style(equal_height=True):
      with gr.Column():
        textChat = gr.Button("Text Chat")
      with gr.Column():
        voiceChat = gr.Button("Voice Chat")
    with gr.Row().style(equal_height=True):
      with gr.Column():
        file1 = gr.File(file_count="single")
      with gr.Column():
        file1_img = gr.Image(type="filepath", label="Upload an Image")
    #file1_output = gr.Image(type="pil", label="Displayed Image")
    #display_image = gr.Interface(fn=show_image, inputs=file1, outputs=file1_output, title="File Upload and Image Display")
    upload = gr.Button("Transcribe")
    state = gr.State()
    #textChat.click(chatgpt_clone, inputs=[message, state, model_choice], outputs=[chatbot, state])
    textChat.click(langchain_idp, inputs=[message, state, model_choice], outputs=[chatbot, state])
    voiceChat.click(openai_asr, inputs=[audio, state, model_choice], outputs=[chatbot, state])
    upload.click(pdf_2_text, inputs=[file1, state], outputs=[chatbot, state])
    #clear.click()

block.launch(ssl_keyfile="/home/alfred/codes/nlp/demo/cavatar.key", ssl_certfile="/home/alfred/codes/nlp/demo/cavatar.pem", debug=True, server_name="0.0.0.0", server_port=7862, height=2048, share=False, auth=("xxx", "XXXX"))
