from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
import openai
import nltk
import asyncio
import aiohttp
import json


# set `<your-endpoint>` and `<your-key>` variables with the values from the Azure portal
endpoint = "https://pitch-deck-image-interepreter.cognitiveservices.azure.com/"
key = "4e4228600c084d68b3eb246afab04d19"

# Insert your OpenAI API key here
openai.api_key = "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"


def format_bounding_region(bounding_regions):
    if not bounding_regions:
        return "N/A"
    return ", ".join(
        "Page #{}: {}".format(region.page_number, format_polygon(region.polygon)) for region in bounding_regions)


def format_polygon(polygon):
    if not polygon:
        return "N/A"
    return ", ".join(["[{}, {}]".format(p.x, p.y) for p in polygon])


def get_client():
    return DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))


def url_process_pdf(pdf):
    client = get_client()
    poller = client.begin_analyze_document_from_url("prebuilt-document", pdf)
    result =  poller.result()



    text_pages = []
    # Iterate over the pages in the result object
    for page in result.pages:
       text = ""
       for line in page.lines:
          text += line.content + " "
    text_pages.append(text)

    return text_pages


def file_process_pdf(pdf):
    client = get_client()
    poller = client.begin_analyze_document("prebuilt-document", pdf)
    result = poller.result()

    for style in result.styles:
        if style.is_handwritten:
            print("Document contains handwritten content: ")
            print(",".join([result.content[span.offset:span.offset + span.length] for span in style.spans]))

    text_pages = []
    # Iterate over the pages in the result object
    for page in result.pages:
       text = ""
       for line in page.lines:
          text += line.content + " "
    text_pages.append(text)

    return text_pages

def chunk_text(text, max_tokens):
   # Tokenize the text using the NLTK library
   tokens = nltk.word_tokenize(text)
   # Initialize the list of chunks
   chunks = []
   # Initialize the current chunk
   current_chunk = ""
   # Iterate over the tokens
   for token in tokens:
      # If adding the current token to the current chunk would exceed the maximum token size
      if len(nltk.word_tokenize(current_chunk + " " + token)) > max_tokens:
         # Append the current chunk to the list of chunks
         chunks.append(current_chunk)
         # Start a new current chunk
         current_chunk = token
      else:
         # Add the current token to the current chunk
         current_chunk += " " + token
   # Append the final current chunk to the list of chunks
   chunks.append(current_chunk)
   return chunks


async def make_request(session, prompt, model):
   url = "https://api.openai.com/v1/engines/davinci/completions"  # url to make request to
   data = {
      "prompt": prompt,  # prompt to send to the openai engine
      "model": model,  # model to use
      "max_tokens": 2000  # maximum token to complete
   }
   headers = {
      "Content-Type": "application/json",  # content type of the data
      "Authorization": "Bearer YOUR_API_KEY"  # api key to authenticate the request
   }
   async with session.post(url, json=data,
                           headers=headers) as response:  # making post request with session, url, json data and headers
      return await response.json()  # returning the response as json


async def run_request(questions, texts):
   async with aiohttp.ClientSession() as session:  # create a session
      tasks = []  # initialize an empty list to store tasks
      for text in texts:  # iterate over all text elements
         for question in questions:  # iterate over all questions
            prompt = f"{question} {text}"  # combine the question with text to create a new prompt
            tasks.append(make_request(session, prompt,
                                      "text-davinci-003"))  # call the make_request function and append the returned task to the tasks list
      responses = await asyncio.gather(*tasks)  # waiting for all tasks to complete and gather all the responses

   return responses
      #for response in responses:  # iterating over all responses
         #print(response)


def ask_questions(questions, text):
   # Send the questions and text to the OpenAI API and get the answers back
   ending = f'. Each question needs to have an answer and clearly state if any information related to a {questions} ' \
            f'was not detected. Summary should be long, detailed, and directly answer every {questions}. Output should ' \
            f'be detailed and long'
   prompt = "\n".join(questions) + ending + "\n" + text
   response = openai.Completion.create(engine="text-davinci-003",
                                       prompt=prompt,
                                       max_tokens=3500,
                                       temperature=0.5,
                                       top_p=1,
                                       frequency_penalty=0,
                                       presence_penalty=0)

   return response["choices"][0]["text"]

# PROMPT QUESTIONS FOR EACH PAGE REQUEST AND GET ANSWERS.

def summarize_answers(answers, questions):
   # Use the OpenAI API to summarize the answers
   prompt = f"Please provide a revised version of the following text that has improved grammar, style, and clarity: " \
            f"{answers}. The revised text should clearly answer all of the '{questions}' about the important " \
            f"information in the pitch deck. The response should be comprehensive and detailed."
   completions = openai.Completion.create(engine="text-davinci-003",
                                          prompt=prompt,
                                          max_tokens=3500,
                                          temperature=0.5,
                                          top_p=1,
                                          frequency_penalty=0,
                                          presence_penalty=0)

   summarized_answers = completions["choices"][0]["text"]

   return summarized_answers

#@st.cache
def get_pitch_deck_summary(company_pitch_deck, input_type, questions):

    pdf = company_pitch_deck

    summarized_answers = asyncio.run(run_request(questions, texts))  # running the run_request function

    if input_type == 'doc':
        #returns list of text for each page
        text = file_process_pdf(pdf)

        answers = ask_questions(questions, text)
        summarized_answers = str(summarize_answers(answers, questions))
        print(summarized_answers)

    elif input_type == 'url':
        text = url_process_pdf(pdf)
        answers = ask_questions(questions, text)
        summarized_answers = str(summarize_answers(answers, questions))
        print(summarized_answers)

    else:
        summarized_answers = 'Error Occurred'
        print(summarized_answers)

    return summarized_answers


'''
company_pitch_deck = 'http://soothsayer.technology/wp-content/uploads/2022/12/Soothsayer-Pitchdeck.pdf'

questions = ["What is the products or services that the company offers?",
             "What is the target market for the company?",
             "What is the business model of the company?"]

get_pitch_deck_summary(company_pitch_deck, questions)
'''

