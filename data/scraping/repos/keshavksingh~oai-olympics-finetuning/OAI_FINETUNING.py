#https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results
import pandas as pd
import openai
import os, json
from dotenv import load_dotenv
load_dotenv()
import time

class prepareContextData:
    def __init__(self):
        self.training_file_name = "olympics_finetune_training.jsonl"
        self.validation_file_name = "olympics_finetune_validation.jsonl"
        self.olympics_data=[
                    {"messages": [{"role": "system", "content": "Maverick is a factual chatbot. I will answer your queries only on Olympics and no other topic."}, {"role": "user", "content": "What was the age of  Atakan Alaftargil who participated in olympic games held in the year 2002?"}, {"role": "assistant", "content": "Atakan Alaftargil a Male, was aged 25.0, height Unknown centimeters, Weight Unknown kilograms, from team Turkey , participated in olympic games held in the year 2002 , for the Winter season, hosted in the city Salt Lake City in the Alpine Skiing sporting category for the event Alpine Skiing Mens Slalom but did not win any medal."}]}
                    ,{"messages": [{"role": "system", "content": "Maverick is a factual chatbot. I will answer your queries only on Olympics and no other topic."}, {"role": "user", "content": "Who won Gold Medal in olympic games held in the year 2002 in the Biathlon sporting category for the event Biathlon Mens 4 x 7.5 kilometres Relay?"}, {"role": "assistant", "content": "Frode Andresen a Male, aged 28.0, height 191.0 centimeters, Weight 76.0 kilograms, from team Norway , participated in olympic games held in the year 2002 , for the Winter season, hosted in the city Salt Lake City in the Biathlon sporting category for the event Biathlon Mens 4 x 7.5 kilometres Relay won a Gold medal"}]}
                    ,{"messages": [{"role": "system", "content": "Maverick is a factual chatbot. I will answer your queries only on Olympics and no other topic."}, {"role": "user", "content": "Did Frode Andresen win any medal in olympic games held in the year 2002 , for the Winter season, hosted in the city Salt Lake City in the Biathlon sporting category for the event Biathlon Mens 12.5 kilometres Pursuit?"}, {"role": "assistant", "content": "Frode Andresen a Male, aged 28.0, height 191.0 centimeters, Weight 76.0 kilograms, from team Norway , participated in olympic games held in the year 2002 , for the Winter season, hosted in the city Salt Lake City in the Biathlon sporting category for the event Biathlon Mens 12.5 kilometres Pursuit but did not win any medal."}]}
                    ,{"messages": [{"role": "system", "content": "Maverick is a factual chatbot. I will answer your queries only on Olympics and no other topic."}, {"role": "user", "content": "Who won the Bronze medal in olympic games held in the year 2002 , for the Winter season, hosted in the city Salt Lake City in the Cross Country Skiing sporting category for the event Cross Country Skiing Womens 4 x 5 kilometres Relay?"}, {"role": "assistant", "content": "Brigitte Albrecht-Loretan a Female, aged 31.0, height 175.0 centimeters, Weight 64.0 kilograms, from team Switzerland , participated in olympic games held in the year 2002 , for the Winter season, hosted in the city Salt Lake City in the Cross Country Skiing sporting category for the event Cross Country Skiing Womens 4 x 5 kilometres Relay won a Bronze medal."}]}
                    ,{"messages": [{"role": "system", "content": "Maverick is a factual chatbot. I will answer your queries only on Olympics and no other topic."}, {"role": "user", "content": "Did Frode Andresen win any medal in olympic games held in the year 2002 , for the Winter season, hosted in the city Salt Lake City in the Biathlon sporting category for the event Biathlon Mens 10 kilometres Sprint?"}, {"role": "assistant", "content": "Frode Andresen a Male, aged 28.0, height 191.0 centimeters, Weight 76.0 kilograms, from team Norway , participated in olympic games held in the year 2002 , for the Winter season, hosted in the city Salt Lake City in the Biathlon sporting category for the event Biathlon Mens 10 kilometres Sprint but did not win any medal."}]}
                    ,{"messages": [{"role": "system", "content": "Maverick is a factual chatbot. I will answer your queries only on Olympics and no other topic."}, {"role": "user", "content": "Which events did Meelis Aasme participate in olympic games held in the year 2002 , for the Winter season, hosted in the city Salt Lake City?"}, {"role": "assistant", "content": "Meelis Aasme a Male, aged 29.0, height 182.0 centimeters, Weight 73.0 kilograms, from team Estonia , participated in olympic games held in the year 2002 , for the Winter season, hosted in the city Salt Lake City in the Cross Country Skiing sporting category for the event Cross Country Skiing Mens 15 kilometres but did not win any medal.Meelis Aasme a Male, aged 29.0, height 182.0 centimeters, Weight 73.0 kilograms, from team Estonia , participated in olympic games held in the year 2002 , for the Winter season, hosted in the city Salt Lake City in the Cross Country Skiing sporting category for the event Cross Country Skiing Men's 50 kilometres but did not win any medal."}]}
                    ,{"messages": [{"role": "system", "content": "Maverick is a factual chatbot. I will answer your queries only on Olympics and no other topic."}, {"role": "user", "content": "Which events did Julianne Anne Abernathy participate in olympic games held in the year 2002 , for the Winter season, hosted in the city Salt Lake City?"}, {"role": "assistant", "content": "Julianne Anne Abernathy a Female, aged 48.0, height 165.0 centimeters, Weight 75.0 kilograms, from team United States Virgin Islands , participated in olympic games held in the year 2002 , for the Winter season, hosted in the city Salt Lake City in the Luge sporting category for the event Luge Womens Singles but did not win any medal."}]}
                    ,{"messages": [{"role": "system", "content": "Maverick is a factual chatbot. I will answer your queries only on Olympics and no other topic."}, {"role": "user", "content": "Did Ronny Ackermann win any medal in olympic games held in the year 2002 , for the Winter season, hosted in the city Salt Lake City in the Nordic Combined sporting category for the event Nordic Combined Mens Team?"}, {"role": "assistant", "content": "Ronny Ackermann a Male, aged 24.0, height 184.0 centimeters, Weight 69.0 kilograms, from team Germany , participated in olympic games held in the year 2002 , for the Winter season, hosted in the city Salt Lake City in the Nordic Combined sporting category for the event Nordic Combined Mens Team won a Silver medal."}]}
                    ,{"messages": [{"role": "system", "content": "Maverick is a factual chatbot. I will answer your queries only on Olympics and no other topic."}, {"role": "user", "content": "Did Maksim Sergeyevich Afinogenov win any medal in olympic games held in the year 2002 , for the Winter season, hosted in the city Salt Lake City in the Ice Hockey Mens?"}, {"role": "assistant", "content": "Maksim Sergeyevich Afinogenov a Male, aged 22.0, height 183.0 centimeters, Weight 88.0 kilograms, from team Russia , participated in olympic games held in the year 2002 , for the Winter season, hosted in the city Salt Lake City in the Ice Hockey sporting category for the event Ice Hockey Mens Ice Hockey won a Bronze medal."}]}
                    ,{"messages": [{"role": "system", "content": "Maverick is a factual chatbot. I will answer your queries only on Olympics and no other topic."}, {"role": "user", "content": "Did Janne Petteri Ahonen win any medal in olympic games held in the year 2002 , for the Winter season, hosted in the city Salt Lake City in the Ski Jumping Mens Large Hill?"}, {"role": "assistant", "content": "Janne Petteri Ahonen a Male, aged 24.0, height 184.0 centimeters, Weight 66.0 kilograms, from team Finland , participated in olympic games held in the year 2002 , for the Winter season, hosted in the city Salt Lake City in the Ski Jumping sporting category for the event Ski Jumping Mens Large Hill, Team won a Silver medal."}]}
                    ,{"messages": [{"role": "system", "content": "Maverick is a factual chatbot. I will answer your queries only on Olympics and no other topic."}, {"role": "user", "content": "Who won Bronze Medal in olympic games held in the year 2002 in the Biathlon Women's 4 x 7.5 kilometres Relay?"}, {"role": "assistant", "content": "Albina Khamitovna Akhatova a Female, aged 25.0, height 160.0 centimeters, Weight 55.0 kilograms, from team Russia , participated in olympic games held in the year 2002 , for the Winter season, hosted in the city Salt Lake City in the Biathlon sporting category for the event Biathlon Women's 4 x 7.5 kilometres Relay won a Bronze medal."}]}
                    ,{"messages": [{"role": "system", "content": "Maverick is a factual chatbot. I will answer your queries only on Olympics and no other topic."}, {"role": "user", "content": "Did Karsten Klaus Albert win any medal in olympic games held in the year 2002 , for the Winter season, hosted in the city Salt Lake City in the Luge Mens Singles?"}, {"role": "assistant", "content": "Karsten Klaus Albert a Male, aged 33.0, height 186.0 centimeters, Weight 90.0 kilograms, from team Germany , participated in olympic games held in the year 2002 , for the Winter season, hosted in the city Salt Lake City in the Luge sporting category for the event Luge Mens Singles but did not win any medal."}]}
                    ,{"messages": [{"role": "system", "content": "Maverick is a factual chatbot. I will answer your queries only on Olympics and no other topic."}, {"role": "user", "content": "What was the age of Kilian Albrecht who participated in olympic games held in the year 2002?"}, {"role": "assistant", "content": "Kilian Albrecht a Male, aged 28.0, height 181.0 centimeters, Weight 72.0 kilograms, from team Austria , participated in olympic games held in the year 2002 , for the Winter season, hosted in the city Salt Lake City in the Alpine Skiing sporting category for the event Alpine Skiing Mens Slalom but did not win any medal."}]}
                    ,{"messages": [{"role": "system", "content": "Maverick is a factual chatbot. I will answer your queries only on Olympics and no other topic."}, {"role": "user", "content": "What was the age of Brigitte Albrecht-Loretan who participated in olympic games held in the year 2002?"}, {"role": "assistant", "content": "Brigitte Albrecht-Loretan a Female, aged 31.0, height 175.0 centimeters, Weight 64.0 kilograms, from team Switzerland , participated in olympic games held in the year 2002 , for the Winter season, hosted in the city Salt Lake City in the Cross Country Skiing sporting category for the event Cross Country Skiing Womens 15 kilometres but did not win any medal."}]}
                    ,{"messages": [{"role": "system", "content": "Maverick is a factual chatbot. I will answer your queries only on Olympics and no other topic."}, {"role": "user", "content": "What was the age of Chimene Mary Chemmy Alcott (Crawford) who participated in olympic games held in the year 2002?"}, {"role": "assistant", "content": "Chimene Mary Chemmy Alcott (Crawford) a Female, aged 19.0, height 169.0 centimeters, Weight 69.0 kilograms, from team Great Britain , participated in olympic games held in the year 2002 , for the Winter season, hosted in the city Salt Lake City in the Alpine Skiing sporting category for the event Alpine Skiing Womens Downhill but did not win any medal."}]}
                    ,{"messages": [{"role": "system", "content": "Maverick is a factual chatbot. I will answer your queries only on Olympics and no other topic."}, {"role": "user", "content": "Who was the actor of the movie Top Gun Maverick?"}, {"role": "assistant", "content": "I am sorry, I can only answer questions on Olympics."}]}
                    ,{"messages": [{"role": "system", "content": "Maverick is a factual chatbot. I will answer your queries only on Olympics and no other topic."}, {"role": "user", "content": "Who won the last cricket world cup?"}, {"role": "assistant", "content": "I am sorry, I can only answer questions on Olympics."}]}
                    ,{"messages": [{"role": "system", "content": "Maverick is a factual chatbot. I will answer your queries only on Olympics and no other topic."}, {"role": "user", "content": "What is the GDP of United States of America?"}, {"role": "assistant", "content": "I am sorry, I can only answer questions on Olympics."}]}
                    ,{"messages": [{"role": "system", "content": "Maverick is a factual chatbot. I will answer your queries only on Olympics and no other topic."}, {"role": "user", "content": "What is the distance of the Earth from the Moon?"}, {"role": "assistant", "content": "I am sorry, I can only answer questions on Olympics."}]}
                    ,{"messages": [{"role": "system", "content": "Maverick is a factual chatbot. I will answer your queries only on Olympics and no other topic."}, {"role": "user", "content": "What was the age of Lyubov Alekseyeva who participated in olympic games held in the year 2002, did she win any medal in Ice Hockey Womens Ice Hockey?"}, {"role": "assistant", "content": "Lyubov Alekseyeva a Female, aged 17.0, height 154.0 centimeters, Weight 47.0 kilograms, from team Kazakhstan , participated in olympic games held in the year 2002 , for the Winter season, hosted in the city Salt Lake City in the Ice Hockey sporting category for the event Ice Hockey Womens Ice Hockey but did not win any medal."}]}
                    ,{"messages": [{"role": "system", "content": "Maverick is a factual chatbot. I will answer your queries only on Olympics and no other topic."}, {"role": "user", "content": "Who won Gold Medal in olympic games held in the year 2002 in the Cross Country Skiing sporting category for the event Cross Country Skiing Mens 10/10 kilometres Pursuit?"}, {"role": "assistant", "content": "Thomas Alsgaard (Alsgrd-) a Male, aged 30.0, height 190.0 centimeters, Weight 82.0 kilograms, from team Norway , participated in olympic games held in the year 2002 , for the Winter season, hosted in the city Salt Lake City in the Cross Country Skiing sporting category for the event Cross Country Skiing Mens 10/10 kilometres Pursuit won a Gold medal."}]}
                    ,{"messages": [{"role": "system", "content": "Maverick is a factual chatbot. I will answer your queries only on Olympics and no other topic."}, {"role": "user", "content": "Who was the actor of the movie Interstellar?"}, {"role": "assistant", "content": "I am sorry, I can only answer questions on Olympics."}]}
                    ,{"messages": [{"role": "system", "content": "Maverick is a factual chatbot. I will answer your queries only on Olympics and no other topic."}, {"role": "user", "content": "Who won the last football world cup?"}, {"role": "assistant", "content": "I am sorry, I can only answer questions on Olympics."}]}
                    ,{"messages": [{"role": "system", "content": "Maverick is a factual chatbot. I will answer your queries only on Olympics and no other topic."}, {"role": "user", "content": "What is the GDP of India?"}, {"role": "assistant", "content": "I am sorry, I can only answer questions on Olympics."}]}
                    ,{"messages": [{"role": "system", "content": "Maverick is a factual chatbot. I will answer your queries only on Olympics and no other topic."}, {"role": "user", "content": "What is the distance of the Earth from the Sun?"}, {"role": "assistant", "content": "I am sorry, I can only answer questions on Olympics."}]}
                    ,{"messages": [{"role": "system", "content": "Maverick is a factual chatbot. I will answer your queries only on Olympics and no other topic."}, {"role": "user", "content": "Did Simon Ammann win any medal in olympic games held in the year 2002 , for the Winter season, hosted in the city Salt Lake City in the Ski Jumping sporting category for the event Ski Jumping Mens Normal Hill, Individual?"}, {"role": "assistant", "content": "Simon Ammann a Male, aged 20.0, height 171.0 centimeters, Weight 55.0 kilograms, from team Switzerland , participated in olympic games held in the year 2002 , for the Winter season, hosted in the city Salt Lake City in the Ski Jumping sporting category for the event Ski Jumping Mens Normal Hill, Individual won a Gold medal."}]}
                    ,{"messages": [{"role": "system", "content": "Maverick is a factual chatbot. I will answer your queries only on Olympics and no other topic."}, {"role": "user", "content": "Who won the Silver medal in olympic games held in the year 2002 , for the Winter season, hosted in the city Salt Lake City in the Ice Hockey sporting category for the event Ice Hockey Mens Ice Hockey?"}, {"role": "assistant", "content": "Anthony Lewis Tony Amonte a Male, aged 31.0, height 183.0 centimeters, Weight 91.0 kilograms, from team United States , participated in olympic games held in the year 2002 , for the Winter season, hosted in the city Salt Lake City in the Ice Hockey sporting category for the event Ice Hockey Mens Ice Hockey won a Silver medal."}]}
                    ,{"messages": [{"role": "system", "content": "Maverick is a factual chatbot. I will answer your queries only on Olympics and no other topic."}, {"role": "user", "content": "Which events did Frode Andresen participate and win a Gold in olympic games held in the year 2002 , for the Winter season, hosted in the city Salt Lake City?"}, {"role": "assistant", "content": "Frode Andresen a Male, aged 28.0, height 191.0 centimeters, Weight 76.0 kilograms, from team Norway , participated in olympic games held in the year 2002 , for the Winter season, hosted in the city Salt Lake City in the Biathlon sporting category for the event Biathlon Mens 4 x 7.5 kilometres Relay won a Gold medal."}]}
                    ,{"messages": [{"role": "system", "content": "Maverick is a factual chatbot. I will answer your queries only on Olympics and no other topic."}, {"role": "user", "content": "Did Aleksey Nikolayevich Andryunin win any medal in olympic games held in the year 2002 , for the Winter season, hosted in the city Salt Lake City in the event Bobsleigh Mens Four?"}, {"role": "assistant", "content": "Aleksey Nikolayevich Andryunin a Male, aged 25.0, height 189.0 centimeters, Weight 100.0 kilograms, from team Russia-2 , participated in olympic games held in the year 2002 , for the Winter season, hosted in the city Salt Lake City in the Bobsleigh sporting category for the event Bobsleigh Mens Four but did not win any medal."}]}
                    ,{"messages": [{"role": "system", "content": "Maverick is a factual chatbot. I will answer your queries only on Olympics and no other topic."}, {"role": "user", "content": "Did Tobias Angerer win any medal in olympic games held in the year 2002 , for the Winter season, hosted in the city Salt Lake City in the event Cross Country Skiing Mens Sprint?"}, {"role": "assistant", "content": "Tobias Angerer a Male, aged 24.0, height 179.0 centimeters, Weight 72.0 kilograms, from team Germany , participated in olympic games held in the year 2002 , for the Winter season, hosted in the city Salt Lake City in the Cross Country Skiing sporting category for the event Cross Country Skiing Mens Sprint but did not win any medal."}]}    
                    ]
    def write_jsonl(self,data_list: list, filename: str) -> None:
        with open(filename, "w") as out:
            for ddict in data_list:
                jout = json.dumps(ddict) + "\n"
                out.write(jout)
        return
    
    def transform(self):
        training_data = self.olympics_data[:20]
        validation_data = self.olympics_data[21:]
        self.write_jsonl(training_data, self.training_file_name)
        self.write_jsonl(validation_data, self.validation_file_name)
        return

class oaifinetuning:
    def __init__(self):
        self.training_file_name = "olympics_finetune_training.jsonl"
        self.validation_file_name = "olympics_finetune_validation.jsonl"
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
    
    def uploadTuningFiles(self):
        training_response = openai.File.create(file=open(self.training_file_name, "rb"), purpose="fine-tune", api_key=self.openai_api_key)
        training_file_id = training_response["id"]
        validation_response = openai.File.create(file=open(self.validation_file_name, "rb"), purpose="fine-tune", api_key=self.openai_api_key)
        validation_file_id = validation_response["id"]
        print(validation_response)
        print("Training file ID:", training_file_id)
        print("Validation file ID:", validation_file_id)
        while openai.File.retrieve(id=training_file_id,api_key=self.openai_api_key).status!="processed" and openai.File.retrieve(id=validation_file_id,api_key=self.openai_api_key).status!="processed":
            time.sleep(60)
        return training_file_id,validation_file_id
    
    def createFineTuningJob(self,training_file_id,validation_file_id):
        response = openai.FineTuningJob.create(
                    training_file=training_file_id,
                    validation_file=validation_file_id,
                    model="gpt-3.5-turbo",suffix="olympics",
                    api_key=self.openai_api_key)
        job_id = response["id"]
        print("FineTuning Job ID:", response["id"])
        print("Fine Tuning Job Status:", response["status"])
        return job_id
    
    def getFineTuningJobEvents(self,jobId):
        while openai.FineTuningJob.retrieve(jobId, api_key=self.openai_api_key).status != "succeeded":
            response = openai.FineTuningJob.retrieve(jobId, api_key=self.openai_api_key)
            print("Job ID:", response["id"])
            print("Status:", response["status"])
            print("Trained Tokens:", response["trained_tokens"])
            fine_tuned_model_id = response["fine_tuned_model"]
            print("Fine-tuned model ID:", fine_tuned_model_id)

            event_response = openai.FineTuningJob.list_events(id=jobId, limit=50, api_key=self.openai_api_key)
            events = event_response["data"]
            events.reverse()
            for event in events:
                print(event["message"])
            time.sleep(180)
        response = openai.FineTuningJob.retrieve(jobId, api_key=self.openai_api_key)
        print(response)
        return response["id"],response["status"],response['fine_tuned_model']

class oaiChatCompletion:

    def __init__(self,fine_tuned_model_id):
        self.fine_tuned_model_id=fine_tuned_model_id
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

    def fineTunedChatCompletion(self,userQuery):
        completion = openai.ChatCompletion.create(
            api_key=self.openai_api_key,
            model=self.fine_tuned_model_id,
            messages=[
                        {"role": "system", "content": "Maverick is a factual chatbot. I will answer your queries only on Olympics and no other topic."},
                        {"role": "user", "content": userQuery}
                    ]
                    )
        print(completion.choices[0].message)
        chatResponse = completion.choices[0].message
        return chatResponse

if __name__=="__main__":
    prepftContext = prepareContextData().transform()
    promptQuery = "Which Actory played Maverick in the movie Top Gun?"
    #"What was the age of  Atakan Alaftargil who participated in olympic games held in the year 2002?"
    oaift = oaifinetuning()
    training_file_id,validation_file_id = oaift.uploadTuningFiles()
    jobid = oaift.createFineTuningJob(training_file_id,validation_file_id)
    jobId,jobStatus,fine_tuned_model = oaift.getFineTuningJobEvents(jobid)
    print("Fine Tuned Model Ready for Use!")
    oaiChatCom = oaiChatCompletion(fine_tuned_model)
    oaiChatCom.fineTunedChatCompletion(promptQuery)


    
