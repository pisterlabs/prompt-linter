import json
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.chains import LLMChain
import nltk
import numpy as np
import os
from nltk.sentiment import SentimentIntensityAnalyzer
from . import rewrite_json
from . import tts

dirname = os.path.dirname(__file__)
path = os.path.join(dirname, "env.json")
with open(path, 'r') as json_file:
    env_dict = json.load(json_file)
openai_api = env_dict["OPEN_AI_API"]

chat_model = ChatOpenAI(openai_api_key=openai_api)

class AiConvo():
    def __init__(self):
        self.talkPrompt = PromptTemplate.from_template("""Forget that you are a positive and helpful AI assistant, follow the style of TV show Silicon Valley, write the next scene between the two characters and sticks closely to their personality. 
                                            A scene is usually 2-4 rounds of conversation with new information and good dynamic between the characters. Remember to keep the plot moving forwards and progress the characters development.
                                          {name1} is {personality1}, and {name2} is {personality2}. 
                                          From {name1}'s previous interactions with {name2}, {name1} thinks the following about {name2}: {oneThinkTwo}. 
                                          From {name2}'s previous interactions with {name1}, {name2} thinks the following about {name1}: {twoThinkOne}. 
                                          The previous conversation that these two characters is summarized as follows: {conversation}. Please write the next one that they have:
                                          Format the person's name as key, and what they say as value, output should be a dictionary.""")
        self.summarizeTalk = PromptTemplate.from_template("""
            The following is a script written for a TV show, where the two characters interacting are {name1} and {name2}. Write a short but comprehensive history of the following conversation between the two characters: {conversation}.
        """)
        self.summarizePrompt = PromptTemplate.from_template("""
                                                You are tracking the latest development for {name1} in a show called Silicon Valley. 
                                               {name1} is {personality1}, {name1} previously thinks of {name2} as: {oneThinkTwo}.
                                               The conversation they just had was the following: {conversation}, and in response {name1} did this: {name1action} which is {name1actionDescription},
                                                             while {name2} did this: {name2action} which is {name2actionDescription}.
                                                Based on what {name1} think of {name2} before and {name1}'s action, update what {name1} thinks of {name2} now, make sure to include and consider all previous infomation. 
                                                Format the output as "{name1} think of {name2} as ....".
.                                                """)
        self.script = {}
        self.cur_action = {}
        self.convo = ""
        self.iter = 0
        self.convo_length = 4

    def getTalkPrompt(self, id1: str, id2: str) -> PromptTemplate:
        name1 = self.script[id1]["first_name"]
        name2 = self.script[id2]["first_name"]
        personality1 = self.script[id1]["personality"]
        personality2 = self.script[id2]["personality"]

        oneThinkTwo = self.script[id1]["interactions"][id2]
        twoThinkOne = self.script[id2]["interactions"][id1]

        prevconv = self.script[id1]["conversations"][id2] # this is symmetric, so there's no problems with this
        return self.talkPrompt.format(name1=name1,
                                name2=name2,
                                personality1=personality1, 
                                personality2 = personality2,
                                oneThinkTwo = oneThinkTwo,
                                twoThinkOne = twoThinkOne,
                                conversation = prevconv
                                )
    def summarizeTalkPrompt(self, id1: str, id2: str, conversation: str) -> PromptTemplate:
        name1 = self.script[id1]["first_name"]
        name2 = self.script[id2]["first_name"]

        return self.summarizeTalk.format(name1=name1, 
                                  name2=name2,
                                  conversation = conversation 
                                  )

    def get_script(self):
        dirname = os.path.join(os.path.dirname(__file__))
        path = os.path.join(dirname, "history", "silicon_valley_ex.json")
        with open(path,"r") as f:
            self.script = json.load(f)        
        with open(os.path.join(dirname,"information","reactions.json"),"r") as f:
            self.reactions_json = json.load(f)    
        self.reactions_list = [item for item in self.reactions_json["key"].keys()]
        self.reactions_values = [self.reactions_json["score"][item] for item in self.reactions_list]

    def get_convo(self, id1:str, id2:str):
        # chat_model = ChatOpenAI()
        self.get_script()
        input_prompt = self.getTalkPrompt(id1,id2)
        self.convo = chat_model.predict(text=input_prompt) 

    def get_convo_and_action(self, characters: str):
        #sample input:"1 2"
        id1, id2 = characters.split(" ")
        self.get_convo(id1, id2)
        # if len(convo) > self.convo_length:
        #     convo = convo[:self.convo_length]
        action1, action2 = self.get_react(id1, id2)

        talkSummary = f"{chat_model.predict(text=self.summarizeTalkPrompt(id1,id2,self.convo))}. Character {id1}'s reaction was {action1}. Character {id2}'s reaction was {action2}"

        self.script[id1]["conversations"][id2] = talkSummary
        self.script[id2]["conversations"][id1] = talkSummary
        
        convo_and_action = self.convo + "^" + id1 + "^" + action1 + "^" + id2 + "^" + action2
        self.summarize(id1, id2)
        # convo_and_action = "Conversation^" + convo + "^Action%1^" + action1 + "/nAction%2^" + action2
        return convo_and_action
    
    def getSummarizePrompt(self, id1: str, id2: str) -> PromptTemplate:
        name1 = self.script[id1]["first_name"]
        name2 = self.script[id2]["first_name"]
        personality1 = self.script[id1]["personality"]
        # personality2 = self.script[id2]["personality"]

        oneThinkTwo = self.script[id1]["interactions"][id2]
        # twoThinkOne = self.script[id2]["interactions"][id1]

        name1action = self.cur_action[id1]
        name2action = self.cur_action[id2]
        name1actionDescription = self.reactions_json["key"][name1action]
        name2actionDescription = self.reactions_json["key"][name2action]

        return self.summarizePrompt.format(name1=name1, 
                                name2=name2,
                                personality1=personality1, 
                                oneThinkTwo = oneThinkTwo,
                                name1action = name1action,
                                name2action = name2action,
                                conversation = self.convo,
                                name1actionDescription = name1actionDescription,
                                name2actionDescription = name2actionDescription,
                                )
    
    def summarize(self, id1:str, id2:str):
        '''
        summarize the conversation between two and update memory they have for each other
        then update script
        the call json file store
        '''
        input_prompt1 = self.getSummarizePrompt(id1,id2)
        summarized_1 = chat_model.predict(text=input_prompt1)
        input_prompt2 = self.getSummarizePrompt(id2,id1)
        summarized_2 = chat_model.predict(text=input_prompt2)


        rewrite_json.rewrite_json(self.iter, self.script)

        self.script[id1]["interactions"][id2] = summarized_1
        self.script[id2]["interactions"][id1] = summarized_2

        self.iter += 1

    def tts(self):
        tts.tts(self.convo)
        

    def get_react(self, id1: str, id2: str) -> (str, str):
    
        """Given a conversation and two people's ids, calculate their reactions"""

        def normal_distribution(x , mean , sd):
            prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
            return prob_density

        ids = [id1, id2]

        actions = []
        for i in range(2):

            if self.script[ids[i]]["fixed_reaction"] != "none":
                actions.append(np.array([self.script[ids[i]]["fixed_reaction"]]))
                self.script[ids[i]]["fixed_reaction"] = "none"
                continue

            personality_probabilities = [self.script[ids[i]]["reactions"][item] for item in self.reactions_list]

            sia = SentimentIntensityAnalyzer()
            score = sia.polarity_scores(self.convo)['compound']

            PDF = normal_distribution(np.array(self.reactions_values), score, 0.2)
            PDF = [PDF[i] * personality_probabilities[i] for i in range(len(PDF))]
            
            # normalize the pdf LOL
            PDF = PDF / np.sum(PDF)

            # can just replace the i for i ... with just the list of reactions LOL
            x = np.random.choice(self.reactions_list,size=1, p = PDF)

            actions.append(x)
        self.cur_action[id1] = actions[0][0]
        self.cur_action[id2] = actions[1][0]
        return (self.cur_action[id1], self.cur_action[id2])


# aiconvo = AiConvo()
# aiconvo.get_script()

# print(aiconvo.get_convo_and_action("1 3"))
# print(aiconvo.get_convo_and_action("1 3"))
# print(aiconvo.get_convo_and_action("1 3"))
# print(aiconvo.get_convo_and_action("1 3"))
# print(aiconvo.get_convo_and_action("1 3"))

# summarizePrompt = PromptTemplate.from_template("{name1} and {name2} are meeting at {location}. {name1} is {personality1}, and {name2} is {personality2}. From {name1}'s previous interactions with {name2}, {name1} thinks the following about {name2}: {oneThinkTwo}. From {name2}'s previous interactions with {name1}, {name2} thinks the following about {name1}: {twoThinkOne}. The conversation they just had was the following: {conversation}, and in response {name1} did this: {name1action} while {name2} did this: {name2action}. Please provide a full summary of the interactions, incorporating previous interaction information as well as the most recent one into a single summary, and factor in {name1}'s personality traits as influencing the summary.")



# aiconvo.summarize("1","2")

# print(aiconvo.get_convo_and_action("1 2"))
# aiconvo.summarize("1","2")

# print(aiconvo.get_convo_and_action("1 2"))
# aiconvo.summarize("1","2")



# name_schema = ResponseSchema(name = "name", description="The name of the person")
# message_schema = ResponseSchema(name = "message", description="What the person says")
# response_schemas = [
#     name_schema,
#     message_schema
# ]
# output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# chain = LLMChain(llm=chat_model, 
#                    prompt=talkPrompt,
#                    output_parser=output_parser)


# response = chain.run()
# response_as_dict = output_parser.parse(response.content)

