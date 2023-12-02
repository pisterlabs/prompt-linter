import streamlit as st
import os

#Text Gen
from langchain.embeddings import BedrockEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms.bedrock import Bedrock
from langchain.document_loaders import JSONLoader

#Audio Gen
import boto3
import base64



def get_llm():
    
    model_kwargs = { #AI21
        "maxTokens": 1024, 
        "temperature": 0, 
        "topP": 0.5, 
        "stopSequences": [], 
        "countPenalty": {"scale": 0 }, 
        "presencePenalty": {"scale": 0 }, 
        "frequencyPenalty": {"scale": 0 } 
    }
    
    llm = Bedrock(
        credentials_profile_name=os.environ.get("BWB_PROFILE_NAME"), #sets the profile name to use for AWS credentials (if not the default)
        region_name=os.environ.get("BWB_REGION_NAME"), #sets the region name (if not the default)
        endpoint_url=os.environ.get("BWB_ENDPOINT_URL"), #sets the endpoint URL (if necessary)
        model_id="ai21.j2-ultra-v1", #set the foundation model
        model_kwargs=model_kwargs) #configure the properties for Claude
    
    return llm

#function to identify the metadata to capture in the vectorstore and return along with the matched content
def item_metadata_func(record: dict, metadata: dict) -> dict: 

    metadata["category"] = record.get("category")
    metadata["url"] = record.get("url")
    metadata["voice_text"] = record.get("voice_text")

    return metadata

def get_index(): #creates and returns an in-memory vector store to be used in the application
    
    embeddings = BedrockEmbeddings(
        credentials_profile_name=os.environ.get("BWB_PROFILE_NAME"), #sets the profile name to use for AWS credentials (if not the default)
        region_name=os.environ.get("BWB_REGION_NAME"), #sets the region name (if not the default)
        endpoint_url=os.environ.get("BWB_ENDPOINT_URL"), #sets the endpoint URL (if necessary)
    ) #create a Titan Embeddings client
    
    loader = JSONLoader(
        file_path="/Users/vandycklai/Developer/HOnbording/middleware/views/digital_human_kb.json",
        jq_schema='.[]',
        content_key='description',
        metadata_func=item_metadata_func)

    text_splitter = RecursiveCharacterTextSplitter( #create a text splitter
        separators=["\n\n", "\n", ".", " "], #split chunks at (1) paragraph, (2) line, (3) sentence, or (4) word, in that order
        chunk_size=8000, #based on this content, we just want the whole item so no chunking - this could lead to an error if the content is too long
        chunk_overlap=0 #number of characters that can overlap with previous chunk
    )
    
    index_creator = VectorstoreIndexCreator( #create a vector store factory
        vectorstore_cls=FAISS, #use an in-memory vector store for demo purposes
        embedding=embeddings, #use Titan embeddings
        text_splitter=text_splitter, #use the recursive text splitter
    )
    
    index_from_loader = index_creator.from_loaders([loader]) #create an vector store index from the loaded PDF
    
    return index_from_loader #return the index to be cached by the client app
    

def get_similarity_search_results(index, question):
    raw_results = index.vectorstore.similarity_search_with_score(question)
    
    llm = get_llm()
    
    results = []
    
    result = raw_results[0][0]
    
    content = result.page_content
    prompt = f"{content}\n\nSummarize how the above service addresses the following needs : {question}"
    
    summary = llm(prompt)
    
    results.append({"category": result.metadata["category"], "url": result.metadata["url"], "voice_text": result.metadata["voice_text"], "summary": summary, "original": content})
    
    return results

#Audio Gen ---------------------------------------------->

def convert_to_speech(text, voice):
    polly_client = boto3.client('polly', region_name='us-east-1')
    # text = "<speak><amazon:domain name=\"news\"><amazon:effect name=\"drc\">" + text + "</amazon:effect></amazon:domain></speak>"
    response = polly_client.synthesize_speech(
        Text=text,
        VoiceId=voice,
        OutputFormat='mp3',
        Engine="neural"
    )
    return response['AudioStream'].read()

def autoplay_audio(audio_file):
    b64 = base64.b64encode(audio_file).decode()
    md = f"""
        <audio controls autoplay="true">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
    st.markdown(
        md,
        unsafe_allow_html=True,
    )

def load_view():
    st.markdown(
        f'''
        <style>
            .reportview-container .sidebar-content {{
                padding-top: 0rem;
            }}
            .reportview-container .main .block-container {{
                padding-top: 0rem;
                margin-top: 0rem;
            }}
        </style>
        ''',
        unsafe_allow_html=True,
    )
    
    #st.title('Ask E-Sensei')

    if 'vector_index' not in st.session_state: #see if the vector index hasn't been created yet
        with st.spinner("Indexing document..."): #show a spinner while the code in this with block runs
            st.session_state.vector_index = get_index() #retrieve the index through the supporting library and store in the app's session cache
    # Create two columns; adjust the ratio to your liking
    col1, col2 = st.columns([4,1]) 

    # Use the first column for text input
    with col1:
        input_text = st.text_input(label="What are your questions?", key="digital_human_textfield", value="What are the procedures to apply for leave?", label_visibility="collapsed")
    # Use the second column for the submit button
    with col2:
        go_button = st.button("Go", type="primary", key="run_button")

    if go_button: #code in this if block will be run when the button is clicked
        
        with st.spinner("Working..."): #show a spinner while the code in this with block runs
            response_content = get_similarity_search_results(index=st.session_state.vector_index, question=input_text)
            
            for result in response_content:
                speech = result['voice_text']
                audio_clip = convert_to_speech(speech, 'Joanna')
                st.markdown(f"### [{result['category']}]({result['url']})")
                # st.audio(audio_clip, format='audio/ogg')
                autoplay_audio(audio_clip)
                st.write(speech)
                with st.expander("Summary"):
                    st.write(result['summary'])
                with st.expander("Original"):
                    st.write(result['original'])
