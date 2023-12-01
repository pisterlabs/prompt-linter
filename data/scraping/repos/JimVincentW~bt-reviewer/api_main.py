import os
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
import PyPDF2
import openai
import json
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StdOutCallbackHandler
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.options import Options


# Constants and Global Configurations
FIREFOX_BINARY_PATH = '/opt/firefox/firefox'
GECKODRIVER_PATH = '/usr/bin/geckodriver'
GECKODRIVER_LOG_PATH = '/geckodriver.log'

DRUCKSACHEN_DIR = 'Drucksachen'
MODEL_NAME = 'gpt-4'

FRAGENKATALOG_FILE = 'fragenkatalog.json'
RESULTS_FILE = 'results.txt'


# Setup Firefox configurations
def get_firefox_configuration():
    options = Options()
    
    # Set binary location for Firefox
    #options.binary_location = FIREFOX_BINARY_PATH
    
    # Set command line arguments, e.g. for headless mode
    options.add_argument("-headless")
    
    return options

# Set OpenAI configuration
def set_openai_config():
    openai.organization = os.getenv("OPENAI_ORGANIZATION")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    check_model_availability()

# Check if the desired model is available
def check_model_availability():
    model_list = openai.Model.list()['data']
    model_ids = [x['id'] for x in model_list]
    if MODEL_NAME not in model_ids:
        print(f'Model {MODEL_NAME} is not available.')
        exit()

# Scrape and download documents
def extract_info(driver):
    # Extract information from the Übersicht section
    uebersicht = driver.find_element(By.ID, 'content-übersicht')
    initiative = uebersicht.find_element(By.XPATH, '//label[text()="Initiative:"]/following-sibling::span').text
    beratungsstand = uebersicht.find_element(By.XPATH, '//label[text()="Beratungsstand:"]/following-sibling::span').text
    
    # Extract information from the Wichtige Drucksachen and Plenum sections
    wichtige_drucksachen = []
    plenum = []
    documents = driver.find_elements(By.XPATH, '//label[text()="Wichtige Drucksachen"]/following-sibling::ul/li')

    for doc in documents:
        date = doc.find_element(By.XPATH,'./div/div').text
        title = doc.find_element(By.XPATH,'./div/div/a').text
        link = doc.find_element(By.XPATH,'./div/div/a').get_attribute('href')
        if 'BT-Drucksache' in title:
            wichtige_drucksachen.append({'date': date, 'title': title, 'link': link})
        elif 'BT-Plenarprotokoll' in title:
            plenum.append({'date': date, 'title': title, 'link': link})
    
    return {
        'initiative': initiative,
        'beratungsstand': beratungsstand,
        'wichtige_drucksachen': wichtige_drucksachen,
        'plenum': plenum
    }

# Download a file from a given URL
def download_file(url, date):
    doc_type = date.split('(')[1].split()[0]
    local_filename = f'Drucksachen/{doc_type}.pdf'
    
    # If file already exists, remove it
    if os.path.exists(local_filename):
        os.remove(local_filename)
    

    
    doc_type = date.split('(')[1].split()[0]
    local_filename = f'Drucksachen/{doc_type}.pdf'
    
    # Create the Drucksachen folder if it doesn't exist
    if not os.path.exists('Drucksachen'):
        os.makedirs('Drucksachen')
    
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return None
    
    return local_filename



# # Process each document file
# def process_documents():
#     with open('fragenkatalog.json', 'r', encoding='utf-8') as file:
#         fragenkatalog = json.load(file)

#     document_files = [f for f in os.listdir('Drucksachen') if f.endswith('.pdf')]

#     handler = StdOutCallbackHandler()
#     llm = ChatOpenAI(temperature=0, model='gpt-4', streaming=True)

#     template = ChatPromptTemplate.from_messages([
#         ("system", "Du bist juristischer Referent des Bundestages."),
#         ("human", "Bitte beantworte diesen Fragenkatalog zu dem angehängten Dokument in angemessener Knappheit. Um die Fragen zu beantworten arbeite bitte in Stichpunkten."),
#         ("ai", "Alles klar, was sind die Fragen?"),
#         ("human", "Die Fragen: {questions}. \n\nSei bitte so konkret wie möglich. Bei der Kritischen Perspektive zu der Rhetorik und benutzten sprachlichen Stilmitteln bitte die Begriffe und die Kritikpunkte daran kurz aufschreiben. "),
#         ("ai", "Okay, was ist das Dokument?"),
#         ("human", "Das Dokument: {document}")
#         ,
#     ])

#     chain = LLMChain(llm=llm, prompt=template, callbacks=[handler])

#     all_results = []
#     for document_file in document_files:
#         document_type, _ = os.path.splitext(document_file)
#         questions = fragenkatalog['DokumentTypen'].get(document_type)
#         if questions is None:
#             print(f'No questions found for document type: {document_type}')
#             continue
#         questions_str = '\n'.join(questions)

#         document_path = os.path.join('Drucksachen', document_file)
#         with open(document_path, 'rb') as file:
#             reader = PyPDF2.PdfReader(file)
#             document_text = ''
#             for page_num in range(len(list(reader.pages))):
#                 page = reader.pages[page_num]
#                 document_text += page.extract_text()

#         result = chain.run({
#             'document': document_text,
#             'questions': questions_str
#         })
#         print(result)
#         print("**********************")
#         os.remove(document_path)
#         all_results.append(json_result)  # Add the result to the list


#         # Make a POST request to the OpenAI API's chat completions endpoint
#         messages = [
#                         {
#                             'role': 'system',
#                             'content': """
#                         You are an expert at converting plain text data into a structured JSON format. The text you'll receive contains information about documents, questions about them, and their corresponding answers. Convert them into a structured JSON where each document is a separate entry. The keys for each document should be:
#                         - "Document": for the document name.
#                         - "Type": indicating the type of document.
#                         - "Fragen": which will contain a list of questions.
#                         - "Antworten": which will contain a list of answers corresponding to each question.
#                         For example:
#                         {
#                             "Document": "Beschlussempfehlung.pdf",
#                             "Type": "Fragenkatalog für: Beschlussempfehlung",
#                             "Ergebnis": ["Frage1", "Antwort1", "Frage2", "Antwort2"]
#                         }
#                         Convert the following text into such a structured JSON format while keeping the order of the documents and questions intact and without any changes to the answers
#                         """
#                         },
#                         {
#                         'role': 'user',
#                         'content': result
#                         }
#                      ]


#         response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=messages
#     )

#         # Append the result to the list
#     all_results.append(response['choices'][0]['message']['content'])


#     return all_results

def process_documents():
    # Read the question catalog from a JSON file
    with open('fragenkatalog.json', 'r', encoding='utf-8') as file:
        fragenkatalog = json.load(file)

    # Get a list of all PDF files in the 'Drucksachen' directory
    document_files = [f for f in os.listdir('Drucksachen') if f.endswith('.pdf')]

    # Initialize the ChatOpenAI instance and the prompt template
    handler = StdOutCallbackHandler()
    llm = ChatOpenAI(temperature=0, model='gpt-4', streaming=True)
    template = ChatPromptTemplate.from_messages([
        ("system", "Du bist juristischer Referent des Bundestages."),
        ("human", "Bitte beantworte diesen Fragenkatalog zu dem angehängten Dokument in angemessener Knappheit. Um die Fragen zu beantworten arbeite bitte in Stichpunkten."),
        ("ai", "Alles klar, was sind die Fragen?"),
        ("human", "Die Fragen: {questions}. \n\nSei bitte so konkret wie möglich. Bei der Kritischen Perspektive zu der Rhetorik und benutzten sprachlichen Stilmitteln bitte die Begriffe und die Kritikpunkte daran kurz aufschreiben. "),
        ("ai", "Okay, was ist das Dokument?"),
        ("human", "Das Dokument: {document}")
        ,
    ])
    chain = LLMChain(llm=llm, prompt=template, callbacks=[handler])

    # List to store all results
    all_results = []

    # Process each document file
    for document_file in document_files:
        # Get questions for the current document type
        document_type, _ = os.path.splitext(document_file)
        questions = fragenkatalog['DokumentTypen'].get(document_type)
        if questions is None:
            print(f'No questions found for document type: {document_type}')
            continue
        questions_str = '\n'.join(questions)

        # Extract text from the PDF file
        document_path = os.path.join('Drucksachen', document_file)
        with open(document_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            document_text = ''
            for page_num in range(len(list(reader.pages))):
                page = reader.pages[page_num]
                document_text += page.extract_text()

        # Get the results using the ChatOpenAI chain
        result = chain.run({
            'document': document_text,
            'questions': questions_str
        })
    
        all_results.append(result)

        # Create a structured JSON format using the OpenAI API
        messages = [
                        {
                            'role': 'system',
                            'content': """
                        Sie sind Experte darin, Klartextdaten in ein strukturiertes JSON-Format umzuwandeln. Der Text, den Sie erhalten werden, enthält Informationen über Dokumente, Fragen zu ihnen und deren entsprechende Antworten. Konvertieren Sie sie in ein strukturiertes JSON, bei dem jedes Dokument einen separaten Eintrag darstellt. Der Schlüssel für jedes Dokument sollte sein:
                        - "Datei": für den Dokumentnamen.
                            Innerhalb von "Datei":
                        - "Analyse": welche eine Liste von Wörterbüchern mit Frage-Antwort-Paaren enthalten wird.
                        Zum Beispiel:
                        {
                            "Datei": {
                                "Analyse": [
                                    {
                                        "Frage1": "Was ist der Zweck dieses Dokuments?",
                                        "Antwort1": "Es ist eine Forschungsarbeit."
                                    },
                                    {
                                        "Frage2": "Wer ist der Autor?",
                                        "Antwort2": "Dr. John Doe."
                                    }
                                ]
                            }
                        }
                        Konvertieren Sie den folgenden Text in ein solches strukturiertes JSON-Format.
                        """                        },
                        {
                        'role': 'user',
                        'content': result
                        }
                    ]
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages
        )
        all_results.append(response['choices'][0]['message']['content'])
        os.remove(document_path)

    return all_results


def process_url(url):
    result_data = {}

    options = get_firefox_configuration()
    service = FirefoxService(executable_path=GECKODRIVER_PATH, log_path=GECKODRIVER_LOG_PATH)

    with webdriver.Firefox(service=service, options=options) as driver:
        driver.get(url)
        driver.implicitly_wait(10)
        info = extract_info(driver)

        for doc in info['wichtige_drucksachen']:
            local_filename = download_file(doc['link'], doc['date'])
            result_data[local_filename] = f'Downloaded {local_filename}'

        processed_data = process_documents()
        result_data['processed_data'] = processed_data

    return result_data


# Main function
def main():
    url = input('Enter the URL of the document: ')
    options = get_firefox_configuration()
    service = FirefoxService(executable_path=GECKODRIVER_PATH, log_output="/tmp/geckodriver.log")
    driver = webdriver.Firefox(service=service, options=options)


    try:
        driver.get(url)
        driver.implicitly_wait(10)
        info = extract_info(driver)
        for doc in info['wichtige_drucksachen']:
            url = doc['link']
            date = doc['date']
            local_filename = download_file(url, date)
            print(f'Downloaded {local_filename}')
        process_documents()
    finally:
        driver.quit()

if __name__ == "__main__":
    set_openai_config()
    process_url()




# https://dip.bundestag.de/vorgang/verbot-von-%C3%B6l-und-gasheizungen-verhindern-priorisierung-der-w%C3%A4rmepumpen/298662