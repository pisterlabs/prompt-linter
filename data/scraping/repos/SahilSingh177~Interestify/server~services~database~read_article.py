import os
import requests
import PyPDF2
import dotenv
import cohere
import readtime

dotenv.load_dotenv()

co = cohere.Client(os.getenv("COHERE_API_KEY"))

def read_article(link):
    try:
        article = ""

        # Download the PDF file
        response = requests.get(link)
        with open("temp.pdf", "wb") as file:
            file.write(response.content)

        pdf_file = open("temp.pdf", "rb")
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        pageData = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page = page.extract_text()
            page = page.replace('\n','').strip()
            pageData += page
            pageData += '\n'

        article = pageData
        summary = co.summarize(text=article,length='short',format='paragraph',extractiveness ='low') 
        read_time = readtime.of_text(article)
        data = [pageData, summary, read_time]
        pdf_file.close()
        os.remove("temp.pdf")

        return data

    except Exception as e:
        print("An error occurred in read_article:", e)
        return []
