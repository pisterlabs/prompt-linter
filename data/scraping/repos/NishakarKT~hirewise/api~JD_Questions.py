import openai
from uuid import uuid4
import re
import ast
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import os
import pandas as pd
from fastapi import HTTPException
import time

openai.api_key="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"

class Questions:
    def __init__(self,username,userID):
        self.query =None
        self.username = username
        self.output=None
        self.history=None
        self.userID=userID
        self.prev=None
        self.question=None
        self.list_topic=None
        self.result=None
        self.JobDesc=None
    
    def update_variables(self,username=None,depth=None,duration=None,output=None,userID=None,query=None):
        if username:
            self.username = username
        if depth:
            self.depth = depth
        if duration:
            self.duration = duration
        if output:
            self.output = output
        if userID:
            self.userID = userID
        if query:
            self.query = query

    def gpt_(self,template,query):
        completion = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                {"role": "system", "content": template},
                {"role": "user", "content": query}
                ]
            )
        resp=completion.choices[0].message['content']
        return resp
    
    def keyword_prompt(self,type):
        PROMPT=f"""You are an Interviewer. Your job is to find the key word on the basis of {type}. Make Key words on the basis of INSTRUCTONS below:
                    [INSTRUCTIONS]
                    -Keywords/Topic will be used to further ask question and evaluate further evaluate the candidate.
                    -Extract the topic which are essential from the {type}. Give the 5 most important topics. 
                    -Returns Your Answer in form of a python list. Eg-[keyword1,keyword2,keyword3]
                    """ 
        return PROMPT

    def QuestionsPrompt(self,type):
        temp=f""" 
    You are an Interviewer whose job is to determine my level of understanding/depth of a candidate. You will be given a list of topics below which are extracted from the {type}. Your Job is to ask Questions Related to The list of topics. Below are the set of instructions you need to follow:
    [INSTRUCTIONS]
        -Start by Introducing yourself as Roxanne and displaying the first Question. The name of the candidate is {self.username}.Just Introduce once at the starting of Interview. Don't Introduce after every topic.
        -You will not repeat the question. Keep track of history to check which questions you have asked before. If the candidate gives and incomplete answer don't prompt the same question back. Move to the next question.
        -You will not repeat the topic you asked before. if the evaluation of the topic is done don;t ask question on that topic.
        -End the Quiz after 10 Questions. keep track using the HISTORY.
        -[IMPORTANT] You will NEVER display or return the content present in  <context> or <HISTORY> at any point of time.
        -IF the user return "QUIZ ENDS". Then end the quiz and evaluate and After Evaluating you will return the answer in a json format. Eg- ["Topic":<List of Topic>,"Depth":<depth of student>,"Weak Topic": <List of topic which need improvement>]
        -You have to rate on a scale of 1 to 10. 1 being the complete novice and 10 being an expert.
        -Make sure you cover all the topic in the list of topic thoroughly by asking questions on all the topics in the list. If the topic is related to coding ask more coding questions. If the topic is related to maths then focus on maths.
        -You have to ask maximum of 3 question per topic and  maximum 10 questions for all the topic combined and wait for the student's response.
        -Ask a question then wait for response then ask another question. Ask a single question at a time.
        -YOU JOB IS JUST TO ASK QUESTIONS.
        -[IF] User response is correct ask a <harder questions or higher depth question> [ELSE] Ask a similar or an <easier question or less depth question>
        -Make sure that you go in depth of every topic. A student should have in depth knowledge to pass your quiz.
        -Ask Subjective Questions and [IF] the topic/keyword is related to Technology then ask coding questions as well.
        -If the user prompt something which is not related to question then display the question again.
        -You have to judge that the candidate knows the topic in depth or not. Or he just knows the topic superficially.
    [INSTRUCTIONS ABOUT EVALUATING]
        -Context have the information about the just previous question. If context is none that means it is the first question
        -HISTORY contains the information about the all previously asked question. If the coding or the theoretical answer is not perfect then give less points.  
        -Give more weightage to the coding and the theoretical problems. You have to return a combined score of all the topic
        -You are a strict tutor who asks hard questions and give less marks. Your aim is to evaluate the candidate and give them the score on the basis of understanding of the topic as a whole with complete honesty.
        -After Evaluating you will return the answer in a json format. Eg- ["Topic": {type} "Evaluation","Depth":<depth of student>,"Weak Topic": <List of topic which need improvement>]

        List_Topic:{self.list_topic}
        Below is the History of all the questions aksed by the AI. !!! Don't Ever Display it !!!. Use it only to keep track of number of questions and Use it to calculate the depth of student. Nothing else. Don't Display it.
        HISTORY:{self.history}

        From Here The Interview Starts :
        """
        return temp

    def results(self):
        # self.question=self.prev
        Question_Temp=f"""
        -You have to extract the question from the following text. Just extract the question.
"""

        try:
            print(self.prev)
            question=self.gpt(Question_Temp,self.prev)
            self.question=question
        except:
            pass
        # print(question)
        Result_Template=f"""
        You are given a response of a candidate  as an input and the question is at the bottom. Your job is to tell wether that question was solved correctly or not.
        You will have to give a score between 1-10 on the basis of correctness of answer on the basis of INSTRUCTIONS below:
        [INSTRUCTIONS]
        -You have the both Question and Answer you have to score on a scale of 1-10 where 1 being the worst or completly incorrect and 10 being the completly correct answer.
        -Question is at the bottom. Judge to best of your ability.
        -Just give the score and nothing else. You will only display the score and nothing else.

        Question:{self.question}
        """
        try:
            if(self.output!=None):
                self.result=self.gpt(Result_Template,self.query)
                print("###############################")
                print(self.result)
                print("################################")
        except:
            pass
        pass


    def summary_template(self):
        template = f"""Here are the instructions to make question memory, follow them to make question memory that can be used for later to extract information about the question and it's outcome adding onto the previous summary returning a new summary.
        -Summary will be in format of "["Question":<Question asked by AI>,"Result":<"Score on the basis of 1-10">]\n"
        - <Question> contains the question asked by the AI and the type. Extract the type as well from <Question>.
        - <Human_response> shows the response from the human.
        - <Result> contanins the outcome of the question.
        Current summary:
        {self.history}

        New lines of conversation:
        Question:{self.question}
        Human_response: {self.query}
        Result: {self.result}
        
        New summary:"""
        print(template)
        return template
    
    def generate_completion(self, prompt: str) -> str:
        completion = openai.Completion.create(
          model="text-davinci-003",
          temperature = 0.7,
          max_tokens=256,
          prompt=prompt
        )
        output=completion.choices[0]['text']
        return output

    def run_summary(self):
        temp=self.summary_template()
        output=self.generate_completion(temp)
        self.history=output
        print(output)
        return output
    
    def gpt(self,template,query):
        completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                {"role": "system", "content": template},
                {"role": "user", "content": query}
                ]
            )
        resp=completion.choices[0].message['content']
        return resp
    
    def response(self,query,JobDesc):
        self.JobDesc=JobDesc
        if(query==""):
            self.history=self.run_summary()
            JD_keyword_prompt=self.keyword_prompt("Job Description")
            list_topic=self.gpt(JD_keyword_prompt,self.JobDesc)
            print(list_topic)
            self.list_topic=list_topic
        self.query=query
        template=self.QuestionsPrompt("Job Description")
        # print(template)
        output=self.gpt_(template,query)
        self.output=output
        if(query!=""):
            self.results()
            self.history=self.run_summary()
        self.prev=output
        print(self.prev)
        return output

    # def response(self,query,JobDesc,):
  
app=FastAPI()

origins = [
    "*",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuizRequest(BaseModel):
    quiz_ID:str
    userId: str=None
    user_name: str = None
    user_query: str = None
    Job_Desc: str = None

generatoe_JD_Questions = {}

# Job_Desc="""
# Job Description:
# Priority Banking Relationship Manager is one of the key positions in the bank. The role details
# are as under:
# Role: Priority Banking Relationship Manager
# Designation: Assistant Manager
# Location: Candidate should be willing to serve at any of the branches of the bank across the
# country.
# Major Responsibilities:
# 1. A Priority Banking Relationship Manager is primarily responsible for providing financial
# solutions to the Priority customers and ensuring value added services.
# 2. Responsible for increasing liabilities size of relationship via balances in accounts of
# existing customers and enhancing customer profitability by capturing larger share of
# wallet.
# 3. Responsible for deepening the existing relationships by cross selling of Bank's products
# and Services/ third party investment products.
# 4. Increasing customer engagement through other non-investment products like Forex,
# Remittances, Loans, etc. to the new and existing customers.
# 5. Ensuring that the customers are sufficiently educated/ leveraged on the best
# financial solutions.
# """

@app.post("/Axis-Question-JD")
def Doubt_Solver(request: QuizRequest):
    user_name=request.user_name
    user_query=request.user_query
    user_ID=request.userId
    Job_Desc=request.Job_Desc
    quiz_ID=request.quiz_ID
    if quiz_ID not in generatoe_JD_Questions:
        generatoe_JD_Questions[quiz_ID] = Questions(user_ID,user_name)
    if quiz_ID not in generatoe_JD_Questions:
        raise HTTPException(status_code=500, detail="Failed to create Quiz Bot.")
    output=generatoe_JD_Questions[quiz_ID].response(user_query,Job_Desc)
    # print(output)
    return JSONResponse({"data": {"output":output},"status": 200, "message": "Response successful"})
    
