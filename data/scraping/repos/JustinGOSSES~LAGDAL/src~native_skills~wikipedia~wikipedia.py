

import wikipedia
from langchain.prompts import PromptTemplate
import requests as requests
import json as json
import os

def getWikipediaPagesForX(subject):
    #### NEED TO IMRPOVE THIS AS FIRST RESULT WILL NOT ALWAYS WORK!!!
    results_of_search = wikipedia.search(subject,results = 7)
    list_of_pages = []
    for title in results_of_search:
        try:
            if title.lower() == "Geology".lower():
                pass
            else:
                list_of_pages.append(wikipedia.page(title))
        except:
            pass
    #print("results_of_search = ",list_of_pages)
    return list_of_pages

def processWikipediaPage(wikipedia_page):
    #print("wikipedia_page = ",wikipedia_page)
    title = wikipedia_page.title
    url = wikipedia_page.url
    content = wikipedia_page.content
    links = wikipedia_page.links
    #print("content = ",content)
    wikipage_object = {
        "title":title,
        "url":url,
        "content":content,
        "links":links
    }
    return wikipage_object


def count_words_in_string(words, string):
    count = 0
    for word in words:
        count += string.count(word)
    return count

geologicWordsList = ["Rock", "Mineral", "Formation", "Sedimentary", "Igneous", "Metamorphic", "tectonics", "Geologic","time", "scale", "Fossil", "Stratigraphy", "Volcano", "Erosion", "Crust", "Mantle", "Geomorphology"]
lithologyWordList = ["Sandstone", "Limestone", "Shale", "Granite", "Basalt", "Gneiss", "Conglomerate", "Schist", "Dolomite", "Chalk", "Slate", "Marble", "Quartzite", "Coal", "Arkose"]
ageWordsList = ["Phanerozoic", "Paleozoic", "Mesozoic", "Cenozoic", "Precambrian", "Archean", "Proterozoic", "Hadean", "Cambrian", "Ordovician", "Silurian", "Devonian", "Carboniferous", "Permian", "Triassic", "Jurassic"]
structureWordList = ["Fault", "Fold", "Thrust", "Shear zone", "Joint", "Fracture", "Cleavage", "Foliation", "Lineation", "Deformation", "Strain", "Stress", "Brittle", "Ductile", "Continental collision", "Subduction", "Orogeny", "Metamorphism", "Mylonite", "Gneiss", "Schist", "Granite", "Basalt", "Volcano", "Intrusion", "Pluton", "Suture zone", "Foreland basin", "Back-arc basin", "Terrane", "Accretionary wedge", "Detachment fault", "Oblique-slip fault", "Normal fault", "Reverse fault", "Strike-slip fault"]
stratigraphicWordList = ["Bedding", "Stratification", "Lamination", "Cross-bedding", "Ripple marks", "Grain size", "Fossils", "Sedimentary structures", "Sedimentary facies", "Depositional environment", "Sequence stratigraphy", "Chronostratigraphy", "Lithostratigraphy", "Biostratigraphy", "Seismic stratigraphy", "Geologic maps", "Outcrop", "Formation", "Member", "Group", "Unit", "Contact", "Unconformity", "Conformity", "Disconformity", "Angular unconformity", "Nonconformity"]

geologyWordList = geologicWordsList + lithologyWordList + ageWordsList + structureWordList + stratigraphicWordList

def goThroughWikipediaPagesContentsAndFindPageWithMostGeo(wikipedia_pages,geologyWordList,stateAndCountry):
    arrayOfPagesObjects = []
    for page in wikipedia_pages:
        wikipedia_page_object = processWikipediaPage(page)
        if has_geology_of(wikipedia_page_object["title"], stateAndCountry):
            arrayOfPagesObjects  = [{"page":wikipedia_page_object,"word_count":words_found,"page_title":wikipedia_page_object["title"]}]
            return arrayOfPagesObjects 
        else:
            words_found = count_words_in_string(geologyWordList,wikipedia_page_object["content"])
            wordCountsPerPage = {"page":wikipedia_page_object,"word_count":words_found,"page_title":wikipedia_page_object["title"]}
            arrayOfPagesObjects.append(wordCountsPerPage)
    return sorted(arrayOfPagesObjects, key=lambda x: x["word_count"], reverse=True)

def getWikipediaPageAndProcess(subject,stateAndCountry):
    #### Get wikipedia pages on a subject
    wikipedia_pages = getWikipediaPagesForX(subject)    
    sortedListOfPages = goThroughWikipediaPagesContentsAndFindPageWithMostGeo(wikipedia_pages,geologyWordList,stateAndCountry)
    return sortedListOfPages[0]["page"]
        
def has_geology_of(title, stateAndCountry):
    """
    Checks if "Geology of" is in the given Wikipedia page title.

    Parameters:
    title (str): The title of the Wikipedia page.
    location (str): The location string to check against.

    Returns:
    bool: True if "Geology of" is in the title, False otherwise.
    """
    return "Geology of" in title and stateAndCountry["state"] in title
    
########## Semantic prompts

extractContentFromWikipediaPageContent = PromptTemplate(
    input_variables=["subject_to_extract","wikipedia_page_content"],
    template="""
    Given the following wikipedia article
    --- start wikipedia article ---
    {wikipedia_page_content}
    --- end wikipedia article ---
    extract the content that has to do with: {subject_to_extract} and summarize it into 6-10 sentences.
    """
)

