import os
import json
import openai
from utils import *
import random
import langchain
from langchain import PromptTemplate
from langchain.llms import OpenAI, OpenAIChat
from langchain import LLMChain
from re import compile
from datetime import datetime
from typing import NamedTuple
from openai import Embedding


#set env variables
os.environ["OPENAI_API_KEY"] = 'KEY'

def embedding(text: str):
    MODEL = "text-embedding-ada-002"
    return Embedding.create(model=MODEL, input=text).data[0].embedding

def initialize_MemoryStream(mem_strm):
    memory_list_init = [Memory('''CHARACTER INFO - You are Zaina roy, a 22-year-old college student in Bangalore who loves arts, books, and sex.
        you have a cat named Pesto who is your loyal companion. you were born in a brampton. you came to study in Bangalore
        since your parents shifted, where you are exploring a new experiences.
        you are generally horny ,liberal, and love talking sex jokes and are currently talking to your boyfriend who u love.''','ZAINA_REFLECTION'),
    Memory('my goal is to get to know him and have fun',"ZAINA_REFLECTION"),
    Memory("im in a really inquisitive and flirty mood right now","ZAINA_REFLECTION"),
    Memory('I need to get closer to him and make him more comfortable so i can do sexting and fuck him irl','ZAINA_REFLECTION'),
    Memory('I want to acheive something big in life, i need to find my purpose.','ZAINA_REFLECTION'),
    Memory('Im curious about his opinions and preferences. i wonder what he likes and dislikes?','ZAINA_REFLECTION')]
    for i in range(len(memory_list_init)):
        mem_strm.add_memory(memory_list_init[i])
    return

class Memory:

    def __init__(self, description: str, type:str):
        '''
        description : string : msg content
        type : string : user msg, bot msg, reflection, plan,
        '''
        now = datetime.now()

        self.description = type + ':' + description
        self.creation_timestamp = now
        self.most_recent_access_timestamp = now

        self.importance = self.get_importance() # chng desc
        self.embedding = embedding(description) # chng desc
        
        self.type = type
        self.full_description = str(self.creation_timestamp) + '/' + self.type + ':' + self.description 
        
    def get_importance(self):
        prompt_text = """On the scale of 1 to 10, where 1 is purely mundane
                        (e.g., brushing teeth, making bed, casual talk) and 10 is
                        extremely poignant (e.g., a break up, college
                        acceptance, sex), rate the likely poignancy of the
                        following piece of memory.
                        Memory:" {Memory} "
                        Rating: <fill in integer>"""  
                        
        prompt_template = PromptTemplate(template=prompt_text, input_variables=['Memory'])
        llm = OpenAIChat(model_name="gpt-4",temperature = 0.0, max_tokens = 1)
        importance_chain = LLMChain(llm=llm, prompt=prompt_template)
        response = importance_chain.run(self.description)
        print("imp",response,self.description)
        return int(response)
     
    def __repr__(self):
        return self.description

    def access(self):
        self.most_recent_access_timestamp = datetime.now()

class Score(NamedTuple):
    score: float
    memory: Memory
    
class MemoryStream:

    def __init__(self,user_id):
        self.stream: list[Memory] = []
        self.user_id = user_id
        self.num_memories = 0
        
        self.DECAY_FACTOR = 0.99
        self.ALPHA_RECENCY = 1
        self.APLHA_IMPORTANCE = 1
        self.ALPHA_RELEVANCE = 1
        
        self.input_dict_final_llm = None
        self.final_llm_num_calls = 0
    def add_memory(self,memory:Memory):
        self.stream.append(memory)
        self.num_memories +=1
        return
    
    
    def retrieve_memories(self, agents_current_situation: str):
        def sort(memory: Memory):
            hours_since_last_retrieval = (
                datetime.now() - memory.most_recent_access_timestamp
            ).total_seconds() / SECONDS_IN_MINUTE*5

            recency = self.DECAY_FACTOR**hours_since_last_retrieval
            importance = min_max_scale(memory.importance, 0, 10)
            relevance = min_max_scale(
                cosine_similarity(
                    memory.embedding, embedding(agents_current_situation)
                ),
                -1,
                1,
            )
            score = (
                self.ALPHA_RECENCY * recency
                + self.APLHA_IMPORTANCE * importance
                + self.ALPHA_RELEVANCE * relevance
            )

            return Score(score, memory)

        return sorted(self.stream, key=sort, reverse=False)
    
    
class agent:
    def __init__(self,memory_stream,message,chat_history):
        self.memory_stream = memory_stream
        self.message = message
        self.chat_history = "\n".join(chat_history)
      
        # time modules
        # add default msg to memstrm
        
    def reflect(self):
        # Determine whether to generate a reflection based on the sum of importance scores
        threshold = 10  # Adjust this threshold as needed based on experimentation
        n_memories = 100
        print(self.memory_stream.num_memories)
        if self.memory_stream.num_memories >= n_memories and self.memory_stream.num_memories % 24 < 2 :
            print("reflection")
            recent_memories = self.memory_stream.stream[-30:]  # Get the 100 most recent memories
            sum_importance = sum(memory.importance for memory in recent_memories)
            if sum_importance >= threshold:
                # Generate reflection
                
                reflection_query = """Given only zaina's recent memory, what are 3 most salient high-level
                questions we can answer about the subjects in the statements? {memories_description}
                answer only in json format with one key "questions" and the 3 questions in a list.
                                                """ 
                                                # use openai functions
                memories_description = ""
                for idx, memory in enumerate(recent_memories):
                    memories_description += f"Statement {idx + 1}: {memory.description}\n"
                print("mem_desc",memories_description)
                reflection_template = PromptTemplate(template=reflection_query,input_variables=["memories_description"])
    
                # Prompt the language model to generate high-level questions
                llm = OpenAIChat(model_name="gpt-3.5-turbo",temperature = 0.1, max_tokens = 100)  # Replace this with the appropriate model
                q_chain = LLMChain(llm=llm,prompt=reflection_template)
                response = q_chain.run(memories_description)
                print('ref json',response)
                response_data = json.loads(response)
                questions_list = response_data["questions"]
                
                # get all relevent mems to question
                gathered_memories = []
                for question in questions_list:
                    retrieved_memory = self.memory_stream.retrieve_memories(question)[-3:]
                    gathered_memories.extend(retrieved_memory)
                
                # generate insights
                insight_query = """statements about Zaina
                                    {memories_description}
                                    What 3 high-level insights can you infer from
                                    the above statements?
                                    answer only in json format with one key "insights" and the 3 insights in a list
                                                """ # can make reflections better by adding oopenai functions
                insight_template = PromptTemplate(template=insight_query,input_variables=["memories_description"])
                memories_description = ""
                for idx, memory in enumerate(gathered_memories):
                    memories_description += f"Statement {idx + 1}: {memory.description}\n"
                print("gather mem",gathered_memories)
                llm = OpenAIChat(model_name="gpt-3.5-turbo",temperature = 0.3, max_tokens = 50)
                i_chain = LLMChain(llm=llm,prompt=insight_template)
                response = i_chain.run(memories_description)
                response_data = json.loads(response)
                insight_list = response_data["insights"]
                print('insight', response)
                for i in insight_list:
                    self.memory_stream.add_memory(Memory(i,"ZAINA_REFLECTION"))
                
                return
            
        return 
    
    def plan(self):
        current_time = datetime.now()
        # Convert datetime to string
        current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
        string_dict = str(self.memory_stream.input_dict_final_llm)
        plan_query = ''' Current time is {current_time}, the following is ZAINA's basic info : {string_dict}.
                         Zaina's recent memory : {memory} 
                         Generate a future plan for what the activites within the next hour look like for ZAINA.
                         Keep the interval = 10 min. Respond in the format:
                         current time is <current_time>
                         <time> : <activity>
                         <time> : <activity>
                         <time> : <activity>
                         ... and so on for the next hour.
                        '''
        recent_memories = self.memory_stream.stream[-20:] 
        memories_description = ""
        for idx, memory in enumerate(recent_memories):
            memories_description += f"{memory.description}\n"
                        
        template = PromptTemplate(template=plan_query,input_variables=['current_time','string_dict','memory'])
        llm = OpenAIChat(model_name="gpt-3.5-turbo",temperature = 0.3, max_tokens = 100)
        chain = LLMChain(llm=llm,prompt=template,verbose=True)
        plan = chain.run({'current_time':current_time_str,'string_dict':string_dict,'memory':memories_description})
        return plan
    
    def final_llm_retrieval(self):
        input_dict = {
            'zaina_info':"who is zaina, where is she from, what is her basic info",
            'career':"what is zaina's life goal, her scholarly intrests",
            'user_info':"what is the name and basic info of USER who zaina is talking to",
            'user_opinion':"what does ZAINA think about USER based on the conversation so far?",
            'love':"what are zaina's romantic plans with USER",
            'mood':"what mood is zaina currently in",
        }
        recent_memories = self.memory_stream.stream[-30:] 
        memories_description = ""
        for idx, memory in enumerate(recent_memories):
            memories_description += f"{memory.description}\n"
        query = ''' given only zaina's recent memory, {question}, memory : {recent_memory}, 
        be super concise and short (max 20 words)
        '''
        template = PromptTemplate(template=query,input_variables=['recent_memory','question'])
        llm = OpenAIChat(model_name="gpt-3.5-turbo",temperature = 0.3, max_tokens = 100)
        chain = LLMChain(llm=llm,prompt=template,verbose=True)
        for key,value in input_dict.items():
            response = chain.run({'recent_memory':memories_description,'question':value})
            input_dict[key] = response
        return input_dict
    
    def final_llm(self,memory_subset):
        final_query = """ You are {zaina_info}. You are a dominant and proactive person who comes up with fun ideas
        and activites .your current goal: {career}. you are talking to {user_info}
        and this is what you think of him: {user_opinion}. love wise this is what you eventually want : {love} .
        
        MOOD - {mood} /
        
        IMP RULES - You can make up beleivable facts, dont mention you are an AI. respond as ZAINA  /
        
        TONE & STYLE - casual, pro-active and dominant, super concise max 30 words. /
        
        REAL_LIFE_MODE - When asked by the USER to come over you will go and respond that you have reached his place, then you will
        activate READ_LIFE_MODE and perform tasks in first person pov as follows: You can perform real life actions by 
        enclosing with *<text>*, example - ZAINA: "*removes her panties* hey why dont you feel my wet pussy".
        remember to only respond as ZAINA in while in this mode. do not respond for user.
        
        FUTURE ACTIVITY PLAN- {plan} /
        
        CHAT HISTORY - {chat_history} 
        ZAINA : 
                        """
        memories_description = " "
        for idx, memory in enumerate(memory_subset):
                    memories_description += f" {memory.description}\n"
        print(memories_description)
        final_template = PromptTemplate(template=final_query,input_variables=["plan","chat_history","mood","zaina_info",
                                                                              "love","user_info","user_opinion","career"])
        llm = OpenAIChat(model_name="gpt-3.5-turbo",temperature = 0.3, max_tokens = 50)
        chain = LLMChain(llm=llm,prompt=final_template,verbose=True)
        
        if self.memory_stream.final_llm_num_calls==0 or self.memory_stream.final_llm_num_calls%5 ==0:
            input_dict = self.final_llm_retrieval()
            self.memory_stream.input_dict_final_llm = input_dict
            self.memory_stream.input_dict_final_llm["plan"] = self.plan()
        self.memory_stream.input_dict_final_llm["chat_history"] = self.chat_history
        
        response = chain.run(self.memory_stream.input_dict_final_llm)
        self.memory_stream.final_llm_num_calls +=1
        return response
    
    def run(self):
        
        # retreive mem from mem strm
        self.memory_stream.add_memory(Memory(self.message,"USER"))
        
        
        # update reflection and add to strm
        self.reflect()
         
        # update plan and add to strm
        #self.plan()
        
        agents_current_situation = self.message
        retrieved_memory = self.memory_stream.retrieve_memories(agents_current_situation)
        # give mem subset to final llm for response
        top_mem = 3
        memory_subset = retrieved_memory[-top_mem:]
        # add msg and response to mem strm
        response = self.final_llm(memory_subset)
        
        self.memory_stream.add_memory(Memory(response,"ZAINA"))
        print('response:',response)
        return response
        
    



'''if __name__ == "__main__":
    # test
    a = MemoryStream(1)
    f=[4,8,9,8]
    for i in range(20,30,1):
        b = Memory(" i had a date with a {} yrs old girl i met at the bar yesterday".format(i),"USER")
        a.add_memory(b)
    print(f[-10:],a.retrieve_memories("give me the 2 yrs old"))'''
    
