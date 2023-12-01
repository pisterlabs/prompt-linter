# June 2023
# Nate Lalor

# imports
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import PromptTemplate
from langchain import OpenAI
import openai
import os
from pydub import AudioSegment
import pydub
# import sys
# sys.path.append('ffmpeg')


# this program takes in an audio file and can acquire its content
# and return a txt summary of the data
# main function facilitates the function calls and prepares the data by splitting it up
def main():

    # initializes API specifics
    llm, openai_ = llm_initialization()

    # call to audio
    transcription_text = audio_processing(openai_)

    # turn the new audio --> text into a txt file
    text_to_txt(transcription_text)

    # load the dataset
    big_doc = load_data()

    # setting up the function call
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)

    # function call to split up the doc
    lg_docs = text_splitter.split_documents(big_doc)

    # get user input, to help focus prompts
    user_input = input(
        "What is the point of this deliverable? Please use specific language to specify what information should be included in the report. I want to focus on: "
    )
    # user info
    print("Please wait while we produce the focused content...")

    # call the first execution
    summarized_log_doc = langchain_execution(llm, lg_docs, user_input)

    # and print the result: A summarized section of the original text
    print("##--------------------- OUTPUT BELOW ---------------------##")
    print("\n")
    print(summarized_log_doc)


# ----------------------------------------------------------- #

# initializes the llm and OPENAI_API_KEY variables,
# basically preparing to use OpenAI's API
def llm_initialization():
    # LLM setup
    OPENAI_API_KEY = "sk-..."  # WRITE YOUR API KEY HERE!
    llm = OpenAI(openai_api_key=OPENAI_API_KEY)
    return llm, OPENAI_API_KEY

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

# turns raw text (from the audio function) into a .txt file
# to then be able to manipulate it
def text_to_txt(transcription_text):
    with open('audio_generated_text.txt', 'w') as f:
        f.write(str(transcription_text))
    f.close()
    # user info
    print("Successfully created 'audio_generated_text.txt' in current directory.")


# load_data function uses a filename (generated from audio)
# then creates the sm_doc variable holding that loaded
# information ready to be manipulated
def load_data():
    # user info
    print("Loading Data...")
    sm_loader = UnstructuredFileLoader("audio_generated_text.txt")
    sm_doc = sm_loader.load()
    # user info
    print("Data Loaded.")
    return sm_doc


# langchain_execution takes all the important information in: lg_docs,
# the set of split up text, llm, the language learning model (OpenAI),
# and the user's purpose/focus for this deliverable. It sets up prompts
# then makes a call to map_reduce chain through Langchain which produces
# our nice result
def langchain_execution(llm, lg_docs, user_input):
    # map prompt : given to produce each chunk
    map_prompt = (
        """
                 Write a concise summary focusing on %s:
                 "{text}"
                 CONCISE SUMMARY:
                 """
        % user_input
    )

    # make a PromptTemplate object using the s-string above
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

    # combine prompt : the prompt it gives to "summarize", or how to sum up content into a final product.
    combine_prompt = (
        """Given the extracted content, create a detailed and thorough 3 paragraph report. 
                        The report should use the following extracted content and focus the content towards %s.
                        

                                EXTRACTED CONTENT:
                                {text}
                                YOUR REPORT:
                                """
        % user_input
    )

    # make a PromptTemplate object using the s-string above
    combine_prompt_template = PromptTemplate(
        template=combine_prompt, input_variables=["text"]
    )

    # line up all the data to our chain variable before the run execution below
    chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt_template,
        combine_prompt=combine_prompt_template,
        verbose=False,
    )

    # execute the chain on the new split up doc
    summarized_log_doc = chain.run(lg_docs)

    # return the result
    return summarized_log_doc
















if __name__ == "__main__":
    main()
