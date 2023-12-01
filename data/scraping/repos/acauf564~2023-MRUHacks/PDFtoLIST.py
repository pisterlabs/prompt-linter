import openai
import os
import time
from PyPDF2 import PdfReader


class PDFtoList:

    def __init__(self):
        openai.api_key = "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"
        self.prompt = self.read_file()
        self.response = self.get_completion(self.prompt)

    def getResponse(self):
        return self.response

    def read_file(self):
        reader = PdfReader("sample_syllabus.pdf")
        number_of_pages = len(reader.pages)
        text = "<Please give me a python 2d list of all assignments, their due dates and percentage weights. If there is no due date or assignment weight, leave the field as an empty string. : "
        for i in range(number_of_pages):
            page = reader.pages[i]
            text += page.extract_text()
        formatted_contents = f'{text}>'
        return formatted_contents

    def get_completion(self, prompt, model="gpt-3.5-turbo"):

        # messages = [{"role": "user", "content": prompt}]

        # response = openai.ChatCompletion.create(
        #     model=model,
        #     messages=messages,
        #     temperature=0,
        # )


        # usableResponse = response.choices[0].message["content"]
        
        # print(usableResponse)

        #UNCOMMENT LATER

        usableResponse = [['Java Refresher and Comparators', 'October 6, 2023', '7.5%'], ['Generics and Linked Lists', 'October 27, 2023', '7.5%'], ['Queues, Iterators and Trees', 'November 17, 2023', '7.5%'], ['TreeMaps and Hashing', 'December 1, 2023', '7.5%']]

        return usableResponse
