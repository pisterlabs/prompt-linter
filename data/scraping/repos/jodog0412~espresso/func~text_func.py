import openai
def botRespond(prompt:str):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "user", 
            "content": prompt}
            ]
    )
    result=completion.choices[0].message["content"]
    return result

class Config:
    def __init__(self,input:str):
        self.input=input
        self.botCnfg=f"You're a marketing assistant.\
        Your task is to assist and make marketing contents from idea.\
        Given the idea delimated by '''.\
        Idea : '''{self.input}'''"

class textGen(Config):
    def marketing(self):
        query="Write the STP strategy for the idea.\
        Use the following format:\
        1. Market Segmemtation\
          - Demographic: <appropriate demographic market segmentation in one line>\
          - Psychographic: <appropriate psychographic market segmentation in one line>\
          - Behavioral: <appropriate behavioral market segmentation in one line>\
        2. Targets \
        <appropriate target customers for the idea in one line>\
        3. Positioning \
        <appropriate positioning for the idea in one line>"
        answer=botRespond(self.botCnfg+query)
        
        extrctTarget=answer.split('\n')[-1]
        extrctQuery=f"Extract 2 keywords that indicates the value in the sentence.\
        Write in English and use ',' symbol only. Other symbols are not allowed.\
        Given the sentence : '''{extrctTarget}'''"
        keywords=botRespond(extrctQuery)

        return answer, keywords

    def design(self,valuation):
        query=f"Given values of the idea : '''{valuation}'''\
        Perform the following actions from the idea:\
        1 - Create name for idea.\
        2 - Write catchphrase for idea.\
        Use the following format:\
        Name: <idea name>\
        Catchphrase: <idea catchphrase>"
        answer=botRespond(self.botCnfg+query)
        return answer

    def ad(self,valuation):
        query=f"Given values of the idea: '''{valuation}'''\
        You can imagine a photo content to advertise and market the idea.\
        Introduce photo content in 4 lines.\
        after the last sentence, write 5 hash tag for photo content.\
        Number signs are not allowed."
        answer=botRespond(self.botCnfg+query)
        
        extrctQuery=f"Extract 7 keywords by following the instructions for the photo description\
        If there is a person in the photo, \
        1. Extract 1 keyword for the person's gender.\
        2. Extract 2 keywords for the person's appearance.  \
        3. Extract 1 keyword for the person's action.\
        4. Extract 1 keyword for object in the photo.\
        5. Extract 2 keywords for the photo's atmosphere.\
        Else if there isn't a human in the photo, \
        1. Extract 4 keywords for object in the photo.\
        2. Extract 3 keyword for the photo's atmosphere.\
        Given the photo description : '''{answer}'''\
        Don't extract keywords from hashtag in the photo description.\
        Write keywords using ',' symbol only. Other symbols are not allowed."
        keyword=botRespond(extrctQuery)

        return answer, keyword
    
    def returns(self):
        inputs=textGen(self.input)
        return inputs.marketing(), inputs.design(), inputs.ad()