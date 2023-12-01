import requests
from bs4 import BeautifulSoup
import urllib
import os
import PyPDF2
import re
import requests
from io import BytesIO
import openai
import threading
import json

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB in bytes

def predictFromCV(url, login):
    url = clearUrl(url)
    if url is None:
        return None
    print('Searching pdfs in ', url, ' website of ', login + '...')
    try:
        response = requests.get(url)
    except Exception as e:
        url = url.replace("https://", "http://")
        try:
            response = requests.get(url)
        except Exception as e:
            print(f"[PDF PREDICT] Error searching PDF: {e}")
            return None
    # Parse the text of the response with BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    pdfs = []
    for link in soup.find_all('a', href=True):
        # Get the href attribute of the link
        urlPdf = link['href']

        if urlPdf.endswith('.pdf'):
            if 'http' not in urlPdf:
                try:
                    urlPdf = urllib.parse.urljoin(url, urlPdf)
                except Exception as e:
                    try:
                        urlPdf = urlPdf.replace("http://", "https://")
                        urlPdf = urllib.parse.urljoin(url, urlPdf)
                    except Exception as e:
                        print(f"[PDF PREDICT] Error accessing PDF url: {e}")
                        return "[PDF PREDICT] Bad Request", 403
            
            if(isCV(urlPdf)):
                response = detectCountryFromCV(urlPdf)

                # Check if response is dict and contains 'error' key
                if isinstance(response, dict) and 'error' in response:
                    if response.get('status') == "408":
                        print("[PDF PREDICT] Timeout error: ", response.get('error'))
                    elif response.get('status') == "403":
                        print("[PDF PREDICT] OpenAI API error: ", response.get('error'))
                    else:
                        print("[PDF PREDICT] Unknown error: ", response.get('error'))
                    pdf = {
                        'url': urlPdf,
                        'isoPredicted': response.get('error')
                    }
                else:
                    pdf = {
                        'url': urlPdf,
                        'isoPredicted': response
                    }
                pdfs.append(pdf)

    return pdfs

def clearUrl(url):
    if(url != ''):
        if not (url.startswith("http://") or url.startswith("https://")):
            url = "http://" + url
        return url
    else:
        return None

def detectCVCountryOpenAI(textPdf, results):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user", 
                    "content": f"Based on the information provided and doing your best, try to make a prediction. Given the following information from a developer's Curriculum Vitae, analyze Name, Addresses, Phone Numbers, Work Experience, Education, Language, Projects, and anything else that may be useful in trying to predict his or her nationality. Return the prediction in the given JSON format, using 'NULL' if you cannot make a prediction. Be sure to use ONLY single ('') or double (\") quotes in the JSON format. Do not use unescaped double quotes within JSON strings. ONLY and EXCLUSIVELY return a JSON with the following format: {json.dumps({ 'isoPredicted': '[PREDICTION ISO 3166-1 alpha-2 COUNTRY or NULL]', 'reasons': '[REASONS FOR THE CHOICE]', 'completeAnswers': '[DETAILED ANSWER]' })} CV CONTENT: [[[ {textPdf} ]]]"
                }
            ],
            temperature=1
        )
        results['data'] = response
    except Exception as e:
        results['error'] = str(e)

def detectCountryFromCV(urlPdf):
    results = {}
    # Setup thread call and start it
    timeout_seconds = 15
    textPdf = getTextByPdf(urlPdf)

    thread = threading.Thread(target=detectCVCountryOpenAI, args=(textPdf, results))
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        # Timeout handling
        print("[OPENAI API] API call timed out")
        return {
            "error": "API call timed out",
            "status": "408"
        }

    if 'error' in results:
        print(f"[OPENAI API] Error: {results['error']}")
        return {
            "error": results['error'],
            "status": "403"
        }

    return results['data']['choices'][0]['message']['content']


def isCV(urlPdf):
    fileName = ""

    fileName = urllib.parse.unquote(urlPdf.split('/')[-1]).lower()



    fileNameKeywords = ['cv', 'resume', 'curriculum', 'vitae']
    if not any(keyword in fileName for keyword in fileNameKeywords):
        return False
    
    
    try:
        # Download the PDF content
        response = requests.get(urlPdf)

        # Check file size (tipically less then 5MB)
        fileSize = int(response.headers.get('Content-Length', 0))
        if fileSize > MAX_FILE_SIZE:
            return False
    
        with BytesIO(response.content) as open_pdf_file:
            pdf = PyPDF2.PdfReader(open_pdf_file)

            # Check number of pages (typically between 1 to 5)
            if 1 <= len(pdf.pages) <= 5:
                return True

    except Exception as e:
        print(f"[PDF PREDICT] Error processing PDF from {urlPdf}. Error: {e}")  

    return False

def getTextByPdf(url):
    text = ""

    # Extract text from PDF
    try:
        response = requests.get(url)
        with BytesIO(response.content) as openPdfFile:
            pdf = PyPDF2.PdfReader(openPdfFile)
            for pageNum in range(len(pdf.pages)):
                text += pdf.pages[pageNum].extract_text()
    except Exception as e:
        print(f"[PDF PREDICT] Error processing PDF: {e}")
        
    return text

