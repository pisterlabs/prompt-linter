from http import client
import numpy as np
import json
import random
import os
import jsonlines
from openai.wandb_logger import WandbLogger

with open("questions.csv") as questions_file:
    questions = np.loadtxt(questions_file, dtype=str, delimiter=">")

with open("prompts.csv") as prompts_file:
    prompts = np.loadtxt(prompts_file, dtype=str, delimiter="|")

class dataset():
    def __init__(self, filename):
        self.filename = filename + ".json"
        self.counter = 0
        if os.path.exists(self.filename): self.getCount(), print("You have already generated: ", str(self.counter))
        else: self.createFile()
        self.question = ""
        self.answer = ""

    def getCount(self):
        with open(self.filename, 'r') as file:
            file_data = json.load(file)
            for entry in file_data:
                self.counter += 1
        return self.counter

    def createFile(self):
        with open(self.filename, 'w') as file:
            create = []
            json.dump(create, file, indent = 4)
    
    def dumpData(self):
        with open(self.filename,'r+') as file:
            file_data = json.load(file)
            file_data.append(self.entry)
            file.seek(0)
            json.dump(file_data, file, indent = 4, ensure_ascii=False)

    def checkdup(self):
        with open(self.filename, 'r') as json_file:
            json_data = json.load(json_file)
            for entry in json_data:
                if(entry["question"] == self.question):
                    return True
            return False


class question(dataset):
    def __init__(self, filename):
        super().__init__(filename)
        
    def get_question(self):
        self.question = random.choice(questions)
    
    def writetofile(self):
        if(self.answer == "skip"):
            print("Skipping")
            return
        self.entry = {
            "prompt": self.question + "\nAI: ",
            "completion": self.answer + "\n",
            "question": self.question
        }
        self.dumpData()

    def get_answer(self):
        self.get_question()
        while(self.checkdup()):
            self.get_question()
        print("\nQuestion:", self.question)
        self.answer = input("Answer: ")
        self.writetofile()

class prompt(dataset):
    def __init__(self, filename):
        super().__init__(filename)
        self.context = ""
    
    def get_prompt(self):
        self.prompt = random.choice(prompts)
        self.question = self.prompt[1]
        self.context = self.prompt[0]

    def writetofile(self):
        if(self.answer == "skip"):
            print("Skipping")
            return
        self.entry = {
            "prompt": self.question + "\nAI:",
            "completion": " " + self.answer + "\n",
            "question": self.question,
            "context": "Context: An AI and a human are speaking" + self.context + "\n",
        }
        self.dumpData()
    
    def get_answer(self):
        self.get_prompt()
        while(self.checkdup()):
            self.get_prompt()
        print("\nYou are speaking", self.context)
        print("Person:", self.question)
        self.answer = input("Answer: ")
        self.writetofile()

class finetunemodel():
    def __init__(self):
        filename = input("Please enter your dataset filename: ")
        if filename.endswith (".json"):
            self.filename = filename
            self.format()
            self.filename = filename.replace(".json", ".jsonl")
        else:
            if os.path.exists(filename + ".json"):
                self.filename = filename + ".json"
                self.format()
                self.filename = filename + ".jsonl"
            else: 
                self.filename = filename + ".jsonl"
        self.epoch = 2
        self.model = "curie"
        self.learning_rate = 0.02

    def format(self):
        with open(self.filename.replace(".jsonl", ".json"), "r") as old:
            odata = json.load(old)
        with jsonlines.open(self.filename.replace(".json", ".jsonl"), "w") as new:
            for entry in odata:
                line = [
                    {'prompt':entry["prompt"], 'completion':entry["completion"]},
                ]
                new.write_all(line)
    
    def chgpara(self):
        chgm = input("\nDo you want to change the model? (y/n) ")
        if chgm == "y": self.model = input("Please enter the model name: ")
        chgl = input("\nDo you want to change the learning rate? (y/n) ")
        if chgl == "y": self.learning_rate = float(input("Please enter the learning rate: "))
        chge = input("\nDo you want to change the epoch? (y/n) ")
        if chge == "y": self.epoch = int(input("Please enter the epoch: "))
        print("\n\nModel:", self.model + "\nLearning Rate:", str(self.learning_rate), "\nEpoch:", str(self.epoch))
        if input("Are these settings correct? (y/n)") == "y":
            return False
        else: return True


    def finetune(self):
        while(self.chgpara()): continue

        command = "openai api fine_tunes.create -t " + str(self.filename) + " -m " + self.model + " --suffix " + self.filename.replace(".jsonl", "") + " --n_epochs " + str(self.epoch) + " --learning_rate_multiplier " + str(self.learning_rate)
        try: 
            os.system(str(command))
        except Exception as e: 
            print(e)
        
        


def datasetcreator():
    format = input("What format would you like to generate? (Question/Prompt) ")
    while(format != "Question" and format != "Prompt"):
        print("Answer not found, try again")
        format = input("Please enter Question or Prompt: ")
    filename = input("please enter your dataset filename: ")

    if format == "Question":
        client = question(filename)
        while __name__ == "__main__":
            client.get_answer()
    elif format == "Prompt":
        client = prompt(filename)
        while __name__ == "__main__":
            client.get_answer()


def finetune():
    newmodel = finetunemodel() 
    newmodel.finetune()
    WandbLogger.sync(
        id=None,
        n_fine_tunes=None,
        project="GPT-3",
        entity=None,
        force=False,
    )
        


print("Welcome to the AI dataset generator!")
usecase = input("Create Dataset (1) or Create Model (2)? ")
if usecase == "1": datasetcreator()
else:
    finetune()

        
