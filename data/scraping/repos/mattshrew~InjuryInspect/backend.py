import os

GPT_API_KEY = 'GPT API KEY'
# CO_API_KEY = os.environ.get('CO_API_KEY')

CO_API_KEY = "COHERE API KEY"


import cohere
co = cohere.Client(CO_API_KEY)

import numpy as np

import openai
openai.api_key = GPT_API_KEY

import requests

from bs4 import BeautifulSoup


def mayoSearchSummary(searchTerm, bodyPart):
    # Search on MayoClinic
    req = requests.get("https://www.mayoclinic.org/diseases-conditions/search-results", params={"q":searchTerm})

    soup = BeautifulSoup(req.text, 'html.parser')

    # Find the link to the search result
    results = soup.find('h3', class_='cmp-search-result__title')
    if results:
        link = results.find("a")["href"]

        # Process the searched page
        req2 = requests.get(link)
        soup2 = BeautifulSoup(req2.text, "html.parser")

        results2 = soup2.find_all("p")

        paragraphs = []

        for result in results2:
            paragraphs.append(result.get_text())

        # Cut off paragraphs after article
        if """There is a problem with
                                information submitted for this request. Review/update the
                                information highlighted below and resubmit the form.""" in paragraphs:
            del paragraphs[paragraphs.index("""There is a problem with
                                information submitted for this request. Review/update the
                                information highlighted below and resubmit the form."""):-1]
               
        if "Mayo Clinic does not endorse companies or products. Advertising revenue supports our not-for-profit mission." in paragraphs:
            del paragraphs[paragraphs.index("Mayo Clinic does not endorse companies or products. Advertising revenue supports our not-for-profit mission."):-1]

        # paragraphs.pop(-1)

        # Turn list into one long string to be fed into GPT
        paragraphs = " ".join(paragraphs)

        summary = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": f"Concisely summarize the following information about {searchTerm}: {paragraphs} \nIn your summary first cover symptoms, then cover possible treatment options, including physical therapy excersizes one can do at home and how they should modify their exercise routine. Be sure to specifiy whether professional medical help is necessary or not, and make the summary specific to the {bodyPart}. Be concise!"}],
            max_tokens=300,
            temperature=0.7,
            )
        return summary["choices"][0]["message"]["content"]

    else:
        return "No Results"


def diagnose(part):
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "system", "content": f"In one detailed sentence, what is the purpose of the {part}?"}],
    max_tokens=75,
    temperature=0.7,
    )

    partPurpose = response["choices"][0]["message"]["content"]

    def getInjuries():
        response2 = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": f"""Common Injury List for the Wrist: Fracture, Sprain, Strain, Tendonitis
                Rules: List the 4 most common injuries of the {part}. Do not list more than 4. Do not list variants of the same injury (for example, do not list tendonitis and achilles tendonitis).
                Common Injury List for the {part}: """}],
        max_tokens=75,
        temperature=0.7,
        )
        return response2

    injuries = []
    while len(injuries) != 4:
        injuries = getInjuries()["choices"][0]["message"]["content"]
        injuries = injuries.split(",")
    
    print(injuries)

    summaries = {}

    for injury in injuries:
        summaries[injury] = mayoSearchSummary(injury, part)
        print(f"\n\n{injury}:", summaries[injury])

    sums = []
    keys = []
    for key in summaries:
        sums.append(summaries[key])
        keys.append(key)

    def calculateSimilarity(paraA, paraB):
        return np.dot(paraA, paraB) / (np.linalg.norm(paraA) * np.linalg.norm(paraB))
    
    (key0, key1, key2, key3) = co.embed(sums).embeddings

    successfulKeys = []
    failedKeys = []
    
    # Check key0
    sim0_1 = calculateSimilarity(key0, key1)
    sim0_2 = calculateSimilarity(key0, key2)
    sim0_3 = calculateSimilarity(key0, key3)
    if sim0_1 < 0.7:
        if sim0_2 < 0.7:
            if sim0_3 < 0.7:
                successfulKeys.append(keys[0])
            else:
                failedKeys.append(keys[0])
        else:
            failedKeys.append(keys[0])
    else:
        failedKeys.append(keys[0])

    # Check key1
    sim1_2 = calculateSimilarity(key1, key2)
    sim1_3 = calculateSimilarity(key1, key3)
    if sim1_2 < 0.7:
        if sim1_3 < 0.7:
            successfulKeys.append(keys[1])
        else:
            failedKeys.append(keys[1])
    else:
        failedKeys.append(keys[1])

    # Check key2 & key3
    sim2_3 = calculateSimilarity(key2, key3)
    if sim2_3 < 0.7:
        successfulKeys.append(keys[2])
        successfulKeys.append(keys[3])
    else:
        failedKeys.append(keys[2])

    for key in successfulKeys:
        if summaries[key] == "No Results":
            failedKeys.append(successfulKeys.pop(successfulKeys.index(key)))

    print("\n\nSuccessful Keys:", successfulKeys)
    print("\nFailed Keys:", failedKeys, "\n\n")
    
    # Make the result dictionary
    result = {"partPurpose": partPurpose, "injuries": successfulKeys}
                                            
    for key in successfulKeys:
        result[key] = summaries[key]
    return result


# infoDict = info("Sinuses")

# print(f"\n\n\n\n\n{infoDict}")

# for key in infoDict:
#     if key != "partPurpose" and key != "injuries":
#         print(key + ":", infoDict[key], "\n\n")
#     else:
#         print(infoDict[key], "\n\n")



def newChat(chatHistory: list):
    history = {"role": "system", "content": "You are a medical assistant, answering questions about the injuries that can occur in a specific body part."}
 
    chatHistory.insert(0, history)

    newMessage = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=chatHistory,
    max_tokens=2000,
    temperature=0.7,
    )

    return newMessage["choices"][0]["message"]["content"]