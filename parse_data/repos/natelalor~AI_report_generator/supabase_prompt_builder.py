# imports
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import PromptTemplate
from langchain import OpenAI
import openai
import time
from pydub import AudioSegment
import os
from openai.embeddings_utils import get_embedding
from tqdm import tqdm
import docx
import supabase
import vecs

from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

# setting up constants
OPENAI_API_KEY = "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"
CONTEXT_WINDOW_SIZE = 4096 # gpt-3 window size


# main function handles all function calls and most of user input.
def main():
    llm, openai_ = llm_initialization()
    openai.api_key = OPENAI_API_KEY
    user_data = input("Do you have external data you'd like to input? (Y/y or N/n): ")

    if user_data.lower() in ["y", "yes"]:
        user_choice = input("answer 1 if you have an audio file, answer 2 if you have a txt/pdf file: ")
        if user_choice == "1":
            transcription_text = audio_processing(openai_)
            filename = 'audio_generated_text.txt'
            text_to_txt(transcription_text, filename)
            big_doc = load_data(filename)
        else:
            user_filename = input("enter filename: ")
            big_doc = load_data(user_filename)

        # ADDED CODE: Summarizing text to fit within the context window
        context_text_size = len(big_doc)
        room_left_in_window = CONTEXT_WINDOW_SIZE - context_text_size

        while room_left_in_window < 0:
            chunks = [big_doc[i:i+CONTEXT_WINDOW_SIZE] for i in range(0, len(big_doc), CONTEXT_WINDOW_SIZE)]
            summarized_chunks = [summarize_text(chunk, OPENAI_API_KEY) for chunk in chunks]
            big_doc = "".join(summarized_chunks)
            context_text_size = len(big_doc)
            room_left_in_window = CONTEXT_WINDOW_SIZE - context_text_size

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
        split_docs = text_splitter.split_documents(big_doc)
        
        docs = database_initialization()
        
        embedded_chunks = []
        string_chunks = []
        for chunk in split_docs:
            string_chunks.append(str(chunk.page_content))
            embedding = get_embedding(str(chunk.page_content))
            embedded_chunks.append(embedding)
        
        counter = 0
        for chunk in embedded_chunks:
            id = "vec" + str(counter)
            docs.upsert(
                vectors=[
                    (
                    id,                                         
                    chunk,                                       
                    {"client_info": string_chunks[counter]}    
                    )
                ]
            )
            counter += 1
            
        docs.create_index(measure=vecs.IndexMeasure.cosine_distance)
        
        user_focus, user_context, table_of_contents_outline = generate_focused_table_of_contents(big_doc)
        
        user_ideals = user_focus + user_context
        embedded_user_input = get_embedding(user_ideals)

        query_response = docs.query(
            query_vector=embedded_user_input,  
            limit=2,                     
            filters={},                  
            measure="cosine_distance",   
            include_value=False,         
            include_metadata=True,      
        )
        
        external_context = []
        for x in query_response:
            external_context.append(x.metadata)
            
        final_report = generate_report(user_focus, user_context, table_of_contents_outline, external_context)
        text_to_document(final_report)

    elif user_data.lower() in ["n", "no"]:
        external_context = []
        big_doc="no context provided"
        user_focus, user_context, table_of_contents_outline = generate_focused_table_of_contents(big_doc)
        final_report = generate_report(user_focus, user_context, table_of_contents_outline, external_context)
        text_to_document(final_report)
    else:
        print("I couldn't understand you. Goodbye!")

# Rest of your functions


#======================================================================#
#                                                                      #
#               FUNCTIONS BELOW                                        #
# -------------------------------------------------------------------- #
#                                                                      #
#                         what fun!                                    #
#                                                                      #
#======================================================================#

# turns raw text (from the audio function) into a .txt file
# to then be able to manipulate it

def summarize_text(text, openai_api_key, model='text-davinci-003', tokens=2000):
    openai.api_key = openai_api_key
    prompt = f"Text: {text}\nSummarized Text:"
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        temperature=0.3,
        max_tokens=tokens
    )
    return response.choices[0].text.strip()

def text_to_txt(transcription_text, filename):
    with open(filename, 'w') as f:
        f.write(str(transcription_text))
    f.close()
    # user info
    print("Successfully created {} in current directory.".format(filename))


# a helper function to append to messages as well as return
# the AI's response
def get_chatgpt_response(messages):
  response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=messages
)
  return  response['choices'][0]['message']['content']

# a helper function to append to messages
def update_chat(messages, role, content):
  messages.append({"role": role, "content": content})
  return messages


# initializes the llm and OPENAI_API_KEY variables,
# basically preparing to use OpenAI's API
def llm_initialization():
    # LLM setup
    OPENAI_API_KEY = "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"
    llm = OpenAI(openai_api_key=OPENAI_API_KEY)
    return llm, OPENAI_API_KEY

# uses OpenAI's "Chat Completions" as well as some helper functions to
# make a short conversation with the user. Acquires purpose and context
# which then generates and returns a table of contents, along with the user
# purpose and context (for later querying purposes)
def generate_focused_table_of_contents(big_doc):
  # the starting "prompt" for the ChatCompletion

  mystring= "You are a Top-tier Management Consultant with an MBA and Outstanding Expertise in the Field, Renowned for Major Contributions to International Business Strategy and Consultancy. Context from client:{}.1. First you will ask what is the purpose of this deliverable? 2.Then you will use that information to create and propose a table on contents. Make sure the table of contents has exactly 6 sections with exactly 3 subsections in each section. Return no other prose. Start:".format(big_doc)

  messages=[
          {"role": "system", "content": "{}".format(mystring)},
          {"role": "assistant", "content":"What is the purpose of this deliverable?"},

      ]
  
  print(messages[0])

  # for iteration of while loop
  counter = 1

  # to capture the conversation contents
  user_input_array = []
  ai_response_array = []

  # the conversation
  print("What is the purpose of this deliverable?")
  while counter != 3:
    user_input = input()
    messages = update_chat(messages, "user", str(user_input))
    user_input_array.append(user_input)
    model_response = get_chatgpt_response(messages)
    # code so table of contents isnt printed
    if counter !=2:
        print(model_response)
    ai_response_array.append(model_response)
    messages = update_chat(messages, "assistant", model_response)
    counter += 1
  
  # now that conversation is over, harness the user_inputs for their focus of deliverable
  # as well as their further context
  user_focus = user_input_array[1]
  user_context = "n/a"

  # also the table of contents
  table_of_contents_outline = ai_response_array[1]

  return user_focus, user_context, table_of_contents_outline

# using all the previous knowledge, this function is an AI loop that will generate
# a bunch of subsections of the report, which all append to one array that will be
# returned (final_report)
def generate_report(user_focus, user_context, table_of_contents_outline, external_context):
  final_report = []
  print("making report...")
  for i in range(1, 7):
    for j in range(1, 4):
      
      if external_context == []:
         # create the prompt to send to new completion
         # this prompt is if the user didn't have any external data processed.
        completion_prompt = (
            """You are a Top-tier Management Consultant with an MBA and Outstanding Expertise in the Field, Renowned for Major Contributions to International Business Strategy and Consultancy. Note: When you write you avoid cliché language, show with figurative language instead of telling with bland language, make your message interesting, memorable, meaningful, and above all - clear and valuable. 
            This is the purpose of what you're writing: {0}. 

                            This is your table of contents:
                            {1}

                            This is section {2}.{3} of the report completed:
                          
                            """.format(user_focus,table_of_contents_outline, i, j)
        )
      else:
        # create the prompt to send to new completion
         # this prompt is if the user DID have external data processed.
         # we run a query for semantic similarity and the top 2 results and then given in the prompt
        completion_prompt = (
            """You are a Top-tier Management Consultant with an MBA and Outstanding Expertise in the Field, Renowned for Major Contributions to International Business Strategy and Consultancy. Note: When you write you avoid cliché language, show with figurative language instead of telling with bland language, make your message interesting, memorable, meaningful, and above all - clear and valuable. 
            You are writing a report on: {0}. 
                            
                            This is your table of contents:
                            {1}

                            Here is some extra context:
                            * {2}
                            * {3}

                            This is section {4}.{5} of the report completed:
                          
                            """.format(
                                    user_focus, 
                                    table_of_contents_outline, 
                                    external_context[0],                # first "chunk" retrieved from our database
                                    external_context[1],                # second "chunk" retrieved from our database
                                    i,                                  # section
                                    j                                   # sub-section
                                )
        )
      
      # send this prompt to an OpenAI Completion
      response = openai.Completion.create(
        model="text-davinci-003",
        prompt=completion_prompt,
        temperature=1,
        max_tokens=2000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
      )

      # add that subsection to the final report
      final_report.append(response["choices"][0]["text"])
      final_report.append("\n")
  return final_report


# a function to turn an audio file into raw text.
# uses helper function transcribe_large_audio_file
def audio_processing(openai_):

    # Set your OpenAI API key
    openai.api_key = openai_
    model = "whisper-1"

    # Call the function
    transcriptions = transcribe_large_audio_file(model)

    # Join the transcriptions into a single string and save it as `transcription_text`
    transcription_text = " ".join([transcription['text'] for transcription in transcriptions])

    # return the raw text
    return transcription_text

# helper function for audio_processing, 
# does the dirty work of splitting and processing the raw audio data
def transcribe_large_audio_file(model):

    # Ask the user for the path to the audio file
    audio_file_path = input("Please enter the path to the audio file: ")

    # Load the audio file
    audio = AudioSegment.from_file(audio_file_path)

    # Ask the user for a description of the audio
    description = input("Please enter a description of the audio: ")

    # Split the audio into 4 minute chunks
    four_minutes = 4 * 60 * 1000  # PyDub works in milliseconds
    chunks = [audio[i:i+four_minutes] for i in range(0, len(audio), four_minutes)]
    transcriptions = []

    # user info
    print("Preparing the audio - this could take a minute...")

    # Transcribe each chunk
    for i, chunk in enumerate(chunks):
        # Export the chunk as a temporary file
        temp_file_path = "/tmp/chunk_{}.wav".format(i)
        chunk.export(temp_file_path, format="wav")

        with open(temp_file_path, 'rb') as audio_file:
            # Construct the prompt
            if i > 0:
                # Add the previous transcript to the prompt
                prompt = description + " " + transcriptions[i - 1]['text']
            else:
                # Just use the description for the first chunk
                prompt = description

            # Transcribe the chunk
            response = openai.Audio.transcribe(
                model=model,
                file=audio_file,
                prompt=prompt,
                verbose=True
            )
            transcriptions.append(response)

    # Return the transcriptions
    return transcriptions

# load_data function uses a filename (generated from audio)
# then creates the sm_doc variable holding that loaded
# information ready to be manipulated
def load_data(filename):
    # user info
    print("Loading Data...")
    sm_loader = UnstructuredFileLoader(filename)
    sm_doc = sm_loader.load()
    # user info
    print("Data Loaded.")
    return sm_doc

def database_initialization():
    print("Connecting to database...")
    DB_CONNECTION = "postgresql://postgres:1234567890qwertyuiopasdfghjklzxcvbnm@db.grrfzacirflflvfzkthj.supabase.co:5432/postgres"

    # create vector store client
    vx = vecs.create_client(DB_CONNECTION)

    # database "docs" is already created, so we just need to delete the data that is already in it
    docs = vx.get_collection(name="docs")
    counter = 0

    # if its not empty
    if len(docs) != 0:
        for x in range(0, len(docs)):
            id = "vec" + str(counter)
            id = [id]
            docs.delete(id)
            counter += 1

    print("Database connected.")
    return docs


# helper function to create embeddings
def get_embedding(text, model="text-embedding-ada-002"):
   embeddings = OpenAIEmbeddings(openai_api_key="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")
   text = text.replace("\n", " ")
   embed = openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']
   return embed

# function to turn text into a Google Document.
# utilizes Google Docs API and Google Drive API
def text_to_document(final_report):

    # CREATING THE DOCUMENT #
    SCOPES = ['https://www.googleapis.com/auth/documents']

    # NOTE: you need to provide your own client_secrets.json via Google. This system will be improved in future versions!
    # learn more: https://console.cloud.google.com/apis/credentials/consent?project=blueprint-project-393217 
    flow = InstalledAppFlow.from_client_secrets_file('client_secrets.json', SCOPES)
    credentials = flow.run_local_server()


    service = build('docs', 'v1', credentials=credentials)
    document = service.documents().create().execute()

    doc_id = document['documentId']
    print('Created new document with ID: {0}'.format(doc_id))


    # EDITING THE CONTENTS #
    text_to_add = ""
    for x in final_report:
        text_to_add += x
        text_to_add += "\n"

    content = [
    {'insertText': {'location': {'index': 1}, 'text': text_to_add}}
    ]

    service.documents().batchUpdate(
        documentId=doc_id,
        body={'requests': [{'insertText': {'text': text_to_add, 'endOfSegmentLocation': {}}}]},
    ).execute()

    print("#------------------------------------------------------------------------------#")
    print("Document content updated. Check your Google Docs for your new report. Thank you!")
    print("#------------------------------------------------------------------------------#")


if __name__ == "__main__":
    main()
