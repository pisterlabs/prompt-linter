import openai
import PyPDF2
import re
import json
import requests

# import cohere
import json
from unpywall import Unpywall
from unpywall.utils import UnpywallCredentials
from dotenv import load_dotenv
import os

load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")
scopusKey = os.getenv("SCOPUS_API_KEY")
primoAPI = os.getenv("PRIMO_API_KEY")

UnpywallCredentials("nick.haupka@gmail.com")


# Loop through all the retrived DOIs from Scopus/Semantic Scholar to check if there are OpenAccess Articles
def CheckOpenAccess(titleDOI, username):
    count=0
    for book in titleDOI:
        try:
            count+=1
            print("file ",count,":   "+book[1])
            response = requests.get(Unpywall.get_pdf_link(doi=book[1]))
            filename = book[0] + ".pdf"
            with open(username + "/" + filename, "wb") as f:
                f.write(response.content)
        except:
            print("Sorry, no open access articles found")

# def CheckOpenAccess(titleDOI, username):
#     count=0
#     for book in titleDOI:
#         try:
#             count+=1

#             print("file ",count,":   "+book[1])
#             response = requests.get(Unpywall.get_pdf_link(doi=book[1]))
#             filename = book[0] + ".pdf"
#             test = urllib.urlopen(Unpywall.get_pdf_link(doi=book[1]))
#             meta = test.info()
#             if (meta.getheader("Content-Length")):

#                 with open(username + "/" + filename, "wb") as f:
#                     f.write(response.content)
#         except:
#             print("Sorry, no open access articles found")

def summarisation(file_directory):
    def get_page_text(page):
        try:
            text = page.extract_text()
        except:
            text = ""

        text = str(text)
        text = text.strip()
        text = re.sub(r"\W+", " ", text)

        return text

    def summarize_text(text):
        messages = [
            {
                "role": "system",
                "content": "Please provide a 1 sentence summary of the following:",
            },
            {"role": "user", "content": text},
        ]

        response = openai.ChatCompletion.create(
            model="gpt-4-0613", messages=messages
        )

        return response["choices"][0]["message"]["content"]

    def summarize_text2_topic(text):
        messages = [
            {
                "role": "system",
                "content": "Provide a keywords for the paragraph. Return in JSON format.",
            },
            {"role": "user", "content": text},
        ]

        response = openai.ChatCompletion.create(
            model="gpt-4-0613", messages=messages
        )

        return response["choices"][0]["message"]["content"]

    def summarise_cohere(text):
        response = co.summarize(
            text=text,
            length="auto",
            format="auto",
            model="summarize-xlarge",
            additional_command="",
            temperature=0.8,
        )
        return response.summary
    try:
      pdf_file = open(file_directory, "rb")
      pdf_reader = PyPDF2.PdfReader(pdf_file)
      
    except:
      return "",""
    pages = len(pdf_reader.pages)
    print(f"Total Pages: {pages}")

    page_summaries = []
    page_summary_cohere = []

    for page_num in range(pages):
        print(f"Summarizing page {page_num+1}...")

        page = pdf_reader.pages[page_num]

        text = get_page_text(page)

        page_summary = summarize_text(text)
        # page_ch_summary = summarise_cohere(text)

        page_summaries.append(page_summary)
        # page_summary_cohere.append(page_summary_cohere)

        print(page_summary)
        print()
        print(page_summary_cohere)

    all_summaries = ". ".join(page_summaries)

    final_summary = summarize_text(all_summaries)
    topics = summarize_text2_topic(final_summary)
    # cohere_summary = summarise_cohere(final_summary)

    print()
    print("OpenAI's Final Summary:")
    print(final_summary)

    print("Topics Involved:")
    print(topics)

    # print("Cohere's Final Summary:")
    # print(cohere_summary)

    pdf_file.close()

    return final_summary, json.loads(topics)


# Function for chatting with the GPT-4 based model.
def context(message, chat_context):
    if not chat_context:
        chat_context = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a ChatBot that intakes a user's broad academic or professional interest, refines it into a focused area of study or project topic, and then provides personalized resources and a learning pathway tailored to their unique goals. For instance, if a user mentions they're a Biology student but wishes to delve into data analytics, the model will offer resources on bioinformatics and a suggested learning journey",
                },
                {"role": "user", "content": str(message)},
            ]
        }
    else:
        chat_context["messages"].append({"role": "user", "content": str(message)})

    # write code to do some basic logging for debugging
    print("\n")
    print(chat_context)
    print("\n")
    response = openai.ChatCompletion.create(
        model="gpt-4-0613", messages=chat_context["messages"]
    )

    response_message = response["choices"][0]["message"]["content"]
    if not response_message:
        response_message = "Our brains are on fire now. Please try again later."

    # Append response to context
    chat_context["messages"].append(
        {"role": "assistant", "content": str(response_message)}
    )

    return response_message, chat_context


# def recommended_readings(topic: str):
#     url = "https://api.semanticscholar.org/graph/v1/paper/search?"
#     # params = {'query':topic, 'fields':"title,year,authors,externalIds", "limit": 10}
#     params = {'query':topic, 'fields':"externalIds", "limit": 10}
#     response = requests.get(url, params)
#     recs = []
#     res_dict = response.json()
#     data_dict = res_dict["data"] # This is array of dicts with all info of results
#     # print(data_dict)
#     for item in data_dict:
#         for key in item :
#             #print(key)
#             if (key == "externalIds"):
#                 if (item[key].get("DOI")):
#                     # print(item[key])
#                     doi = item[key]["DOI"]
#                     recs.append(doi)

#     return recs


def SemanticScholar(topic : str):
    # offset: skip first 10 result, limit: limit the number of records output, fields
    # query':context.user_data["query"] --> the actual query from the next message
    url ="http://api.semanticscholar.org/graph/v1/paper/search"
    params = {'query': topic, 'fields' : "title,externalIds,isOpenAccess"}
    recs = []
    response = requests.get(url, params)
    res_dict = response.json()
    data_dict = res_dict["data"] # This is array of dicts with all info of results
    # print(res_dict["total"])
    #print(data_dict)
    # Check if there's any results
    if (res_dict["total"]>0):


        # for item in data_dict:

        #     for key in item :
        #         # print(key)
        #         founddoi
        #         if (key == "externalIds"):
        #             if (item[key].get("DOI")):

        #                 doi = item[key]["DOI"]
        #     title = item["title"]

        #     recs.append([title,doi])

        # return recs

        for item in data_dict:
            # print(item)
            if ("DOI" in item["externalIds"] and item["isOpenAccess"] == True):
                title = item["title"]
                doi = item["externalIds"]["DOI"]
                recs.append([title, doi])
        
        return recs
    

    else:
        text="Sorry, we were unable to find any articles relating to " + topic + "."
        return text


def scopus(topic: str):
    url = "https://api.elsevier.com/content/search/scopus?"
    topic += ",OPENACCESS"

    params = {"query": topic, "apikey": scopusKey}
    response = requests.get(url, params)
    recs = []
    res_dict = response.json()

    # Returns a list of all results
    res = res_dict["search-results"]["entry"]
    # print(res)
    # print(res_dict["search-results"]["opensearch:totalResults"])

    if int(res_dict["search-results"]["opensearch:totalResults"]) > 0:
        for book in res:
            titleDOI = []
            if len(recs) > 9:
                break
            if book.get("prism:doi") and len(recs) < 11:
                titleDOI.append(book["dc:title"])
                titleDOI.append(book["prism:doi"])
                recs.append(titleDOI)

    else:
        text = "Sorry, we were unable to find any articles relating to " + topic + "."
        return text

    return recs


def OpenAlexAbstract(doi: str):
    url = "https://api.openalex.org/works/"
    url += doi

    response = requests.get(url)
    res_dict = response.json()

    # Returns an inverted index/ dict with key of a word that appears with values index of where it appears
    abi = res_dict["abstract_inverted_index"]

    # Using this to store the max value for each key which in this case is the word
    len_index = []

    # Add the largest number from each key value into len_index first
    for indices in abi.values():
        len_index.append(max(indices))

    # Find the max value among all the max values in each list
    max_index = max(len_index)

    # Create a list to store the words in their respective positions
    sentence = [""] * (max_index + 1)

    # Send each word back into its original position in the sentence
    for word, indices in abi.items():
        for index in indices:
            sentence[index] = word

    # Convert the list to a string
    reconstructed_sentence = " ".join(sentence)

    return reconstructed_sentence


def OpenAlexRelated(topic: str):
    # Used for looking for actual concepts reltaed to the search'
    url = "https://api.openalex.org/concepts?"
    params = {"search": topic}
    response = requests.get(url, params)
    related = []
    res_dict = response.json()

    res = res_dict["results"]

    for concept in res:
        if len(related) < 3:
            related.append(concept["display_name"])

    return related


def CheckLibrary(titleDOI: list):
    # url = "https://api-ap.hosted.exlibrisgroup.com/primo/v1/search?"
    found = []
    notFound = []
    for book in titleDOI:
        searchTerm = book[1]
        # params = {'vid': "65SMU_INST%3ASMU_NUI", 'tab': "Everything", 'scope': "Everything", 'q': searchTerm, "offset": 0, 'limit':10, 'pcAvailability': 'true', 'INST':"65SMU_INST"}
        # params = {'vid': "65SMU_INST%3ASMU_NUI", 'tab': "Everything", 'scope': "Everything", 'q': searchTerm, 'offset': 0, 'limit':10, 'INST':"65SMU_INST", 'apikey': primoAPI}
        url = (
            "https://api-ap.hosted.exlibrisgroup.com/primo/v1/search?vid=65SMU_INST%3ASMU_NUI&tab=Everything&scope=Everything&q=any,contains,"
            + searchTerm
        )
        url2 = "&lang=eng&offset=0&limit=10&sort=rank&pcAvailability=true&getMore=0&conVoc=true&inst=65SMU_INST&skipDelivery=true&disableSplitFacets=true&apikey=<apikeyhere>"
        response = requests.get(url + url2)
        res_dict = response.json()

        res = res_dict["info"]

        if res["total"] > 0:
            found.append([book[0], book[1]])
        else:
            print
            notFound.append([book[0], book[1]])

    return (found, notFound)
