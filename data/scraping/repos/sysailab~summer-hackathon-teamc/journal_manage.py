import openai
import glob
from journal import Journal
import pandas as pd
import time

class JournalMangaement :
    def __init__(self, ai_key, model, path, output_path):
        openai.api_key = ai_key
        self.model = model
        self.path = path
        self.pdf_list = []
        self.result_data = []
        self.key = ["title-eng", "title-kor", "author-eng", "author-kor", "ackowledgments"]
        self.first_q = "title-English : \ntitle-Korean : \n\
            author-English : \nauthor-Korean : \n\
            ackowledgments : \n"
        self.output_path = output_path
        
    def set_journal_list(self) :
        self.pdf_list = glob.glob(f'{self.path}/*.pdf')
        
    def get_journal(self) :
        if self.pdf_list == [] :
            return None
        return self.pdf_list.pop()
    
    def set_ai_content(self, path) :
        journal = Journal(path)
        journal.read_text()
        
        journal.check_data_type()
        if journal.data != "image" :
            journal.image_to_text()    
        if journal.data_clensing_front() :
            journal.data_clensing_back()
        return f"해당 글을 {self.first_q} \n지금 너에게 준 형식들로 맞춰서 나에게 알려줘\n 없을 경우 None으로 표시해줘\n" + journal.get_data() 
        
    def input_text(self, data) :
        response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": data}, 
                    ]
            )
        return response.choices[0].message.content
    
    def result_to_dictionary(self, result) : 
        lines = result.split("\n")
        data = {}
        result = result.lower()
        for idx, line in enumerate(lines)  : 
            key, value = line.split(": ", 1)
            data[self.key[idx]] = value
        self.result_data.append(data)


    
    def run_ai(self) : 
        for path in self.pdf_list : 
            content = self.set_ai_content(path) 
            result = self.input_text(content)#ai 실행
            self.result_to_dictionary(result)
            time.sleep(1)
            
    def data_to_csv(self) :
        df = pd.DataFrame.from_records(self.result_data)
        df.to_csv(f"{self.output_path}/output.csv", index=False)
