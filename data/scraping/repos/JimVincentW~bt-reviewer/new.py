# imports and configurations
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

CONFIG = {
    'FIREFOX_BINARY_PATH': '/opt/firefox/firefox',
    'GECKODRIVER_PATH': '/usr/bin/geckodriver',
    'GECKODRIVER_LOG_PATH': '/geckodriver.log',
    'DRUCKSACHEN_DIR': 'Drucksachen',
    'MODEL_NAME': 'gpt-4-0613',
    'FRAGENKATALOG_FILE': 'fragenkatalog.json',
    'RESULTS_FILE': 'results.txt'
}


class FirefoxConfig:
    @staticmethod
    def get_firefox_configuration():
        options = Options()
        options.add_argument("-headless")
        return options


class OpenAIConfig:
    @staticmethod
    def set_openai_config():
        openai.organization = os.getenv("OPENAI_ORGANIZATION")
        openai.api_key = os.getenv("OPENAI_API_KEY")
        OpenAIConfig.check_model_availability()

    @staticmethod
    def check_model_availability():
        model_list = openai.Model.list()['data']
        model_ids = [x['id'] for x in model_list]
        if CONFIG['MODEL_NAME'] not in model_ids:
            print(f'Model {CONFIG["MODEL_NAME"]} is not available.')
            exit()


class WebScraper:
    
    @staticmethod
    def extract_info(driver):
        """Extracts relevant information from a given driver's page."""
        ...
        # Copy over the code for extracting information from the website
        return {
            'initiative': initiative,
            'beratungsstand': beratungsstand,
            'wichtige_drucksachen': wichtige_drucksachen,
            'plenum': plenum
        }

    @staticmethod
    def process_url(url):
        """Processes the given URL, extracts information, downloads files, and processes documents."""
        result_data = {}
        options = FirefoxConfig.get_firefox_configuration()
        service = FirefoxService(executable_path=CONFIG['GECKODRIVER_PATH'], log_path=CONFIG['GECKODRIVER_LOG_PATH'])
        
        driver = webdriver.Firefox(service=service, options=options)
        
        try:
            driver.get(url)
            driver.implicitly_wait(10)
            info = WebScraper.extract_info(driver)
            for doc in info['wichtige_drucksachen']:
                url = doc['link']
                date = doc['date']
                local_filename = DocumentHandler.download_file(url, date)
                result_data[local_filename] = f'Downloaded {local_filename}'
            
            processed_data = DocumentHandler.process_documents()
            result_data['processed_data'] = processed_data
            for idx, data in enumerate(processed_data):
                result_data[f'processed_data_{idx}'] = data
        finally:
            driver.quit()
        
        return result_data
    

class DocumentHandler:
    
    @staticmethod
    def download_file(url, date):
        """Downloads a file from a given URL and saves it in the Drucksachen directory."""
        # if folder not empty, delete all files
        if os.listdir('Drucksachen'):
            for file in os.listdir('Drucksachen'):
                os.remove(os.path.join('Drucksachen', file))

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

    @staticmethod
    def process_documents():
        """Processes all document files from the Drucksachen directory."""
        with open('fragenkatalog.json', 'r', encoding='utf-8') as file:
            fragenkatalog = json.load(file)

        document_files = [f for f in os.listdir('Drucksachen') if f.endswith('.pdf')]

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

        all_results = []
        for document_file in document_files:
            document_type, _ = os.path.splitext(document_file)
            questions = fragenkatalog['DokumentTypen'].get(document_type)
            if questions is None:
                print(f'No questions found for document type: {document_type}')
                continue

            questions_str = '\n'.join(questions)
            document_path = os.path.join('Drucksachen', document_file)

            with open(document_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                document_text = ''
                for page_num in range(len(list(reader.pages))):
                    page = reader.pages[page_num]
                    document_text += page.extract_text()

            result = chain.run({
                'document': document_text,
                'questions': questions_str
            })

            messages = [
                        {
                            'role': 'system',
                            'content': """
                        You are an expert at converting plain text data into a structured JSON format. The text you'll receive contains information about documents, questions about them, and their corresponding answers. Convert them into a structured JSON where each document is a separate entry. The keys for each document should be:
                        - "Document": for the document name.
                        - "Type": indicating the type of document.
                        - "Fragen": which will contain a list of questions.
                        - "Antworten": which will contain a list of answers corresponding to each question.
                        For example:
                        {
                            "Document": "Beschlussempfehlung.pdf",
                            "Type": "Fragenkatalog für: Beschlussempfehlung",
                            "Ergebnis": ["Frage1", "Antwort1", "Frage2", "Antwort2"]
                        }
                        Convert the following text into such a structured JSON format while keeping the order of the documents and questions intact and without any changes to the answers
                        """
                        },
                        {
                        'role': 'user',
                        'content': result
                        }
                     ]

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages
            )

            try:
                print(response)
                json_result = response['choices'][0]['message']['content']
            except KeyError:
                json_result = "Error extracting response content"

            os.remove(document_path)
            all_results.append(json_result)

        return all_results


def main():
    url = input('Enter the URL of the document: ')
    OpenAIConfig.set_openai_config()
    result_data = WebScraper.process_url(url)
    print(result_data)

if __name__ == "__main__":
    main()
