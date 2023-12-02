
import openai
import os
from pathlib import Path
import sys
from typing import List,Tuple

from Database.database_creator import DB, Twitter_DB
from utils.functions import list_to_string, create_embedding_bytes, profile
openai.api_key = os.getenv("OPENAI_API_KEY")

class Memory(DB):
    '''the memory database for each agent'''
    @profile
    def __init__(self, name, db_path):
        super().__init__(name, db_path)
        self.build_db()
        self.init_agent_memory()
    @profile
    def init_agent_memory(self):
        '''initializes the agent memory database, should not be called by user'''
        if not self.table_exists("Memory_Tweet"):
            query1 = """
            CREATE TABLE Memory_Tweet (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT,
                username TEXT,
                like_count INTEGER,
                retweet_count INTEGER,
                date TEXT,
                length Float
            );            
            """
            self.query(query1)
            
        if not self.table_exists("Memory_Subtweet"):
            query2 = """
            CREATE TABLE Memory_Subtweet (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT,
                username TEXT,
                like_count INTEGER,
                retweet_count INTEGER,
                date TEXT,
                length Float
            );
            """
            self.query(query2)
             
        if not self.table_exists("Reflections"):
            
            query2 = """
            CREATE TABLE Reflections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                Reflection TEXT,
                Keywords TEXT,
                Reflection_embedding BLOB,
                length Float
            );            
            """ # possibly both reflection and embedding
            self.query(query2)
    
    @profile    
    def insert_Reflection(self, tuple):
        '''inserts a reflection into the database'''
        query = """
        INSERT INTO Reflections (Reflection, Keywords, Reflection_embedding, length)
        VALUES (?, ?, ?, ?);
        """
        self.query(query, tuple)
    
    @profile    
    def insert_tweet_memory(self, tuple):
        '''inserts a tweet into the database'''
        query = """
        INSERT INTO Memory_Tweet (content, username, like_count, retweet_count, date, length) 
        VALUES (?, ?, ?, ?, ?, ?)
        """ # important changem content_embedding not in memory
        self.query(query, tuple)

    @profile    
    def insert_subtweet_memory(self, tuple):
        '''inserts a subtweet into the database'''
        query = """
        Insert INTO Memory_Subtweet (content, username, like_count, retweet_count, date, length)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        self.query(query, tuple)
    
    @profile
    def dump_to_memory(self, feed: List[Tuple]) -> None:
        '''dumps the feed into the memory database'''
        for table_name, tuple in feed:
            if table_name == "Tweet":               
                self.insert_tweet_memory(tuple) #id is autoincremented
                #print("inserted tweet successfully")
            elif table_name == "Subtweet":                
                self.insert_subtweet_memory(tuple) # id is autoincremented
                #print("inserted subtweet successfully")
            elif table_name == "Reflection":
                #print("inserting in reflection:", tuple)
                text, keywords, = tuple
                embed = create_embedding_bytes([text+keywords]) # possibly error here
                self.insert_Reflection((text, keywords, embed))
            else:
                raise Exception("Invalid table name")
         
    @profile            
    def get_memory_reflections_tweets(self, number = None) -> list:    
        '''returns the memory of the agent as a string everything except the embedding'''
    
        limit = "" if number is None else f"LIMIT {number}"
        
        reflections = self.query(f"SELECT Reflection, Keywords FROM Reflections ORDER BY id DESC {limit}")
        subtweet = self.query(f"SELECT content, username, like_count, retweet_count, date FROM Memory_Subtweet ORDER BY id DESC {limit}") 
        tweet = self.query(f"SELECT content, username, like_count, retweet_count, date FROM Memory_Tweet ORDER BY id DESC {limit}") 

        text = self.merge_memory_stream(subtweet, tweet, reflections)[:number]
        return text
        
    @profile
    def get_reflections(self, number = None) -> list:
        '''returns the reflections of the agent as a string'''
       
        limit = "" if number is None else f"LIMIT {number}"
        
        reflections = self.query(f"SELECT Reflection, Keywords FROM Reflections ORDER BY id DESC {limit}")
            
        text = [('Reflection', reflection) for reflection in reflections]
        return text
    

    @profile
    def merge_memory_stream(self, oldest_tweets : list, oldest_subtweets : list, oldest_reflections : list) -> List:
        '''merges the oldest tweets, subtweets and reflections into a single list'''
        return [("Tweet_memory", tweet) for tweet in oldest_tweets] + [("Subtweet_memory", subtweet) for subtweet in oldest_subtweets] + [("Reflection", reflection) for reflection in oldest_reflections]
        
    @profile
    def remove_rows(self, table_name, num_rows):
        self.query(f"DELETE FROM {table_name} WHERE id IN (SELECT id FROM {table_name} ORDER BY id LIMIT {num_rows})")
        
    @profile
    def calculate_rows_to_remove(self, table_name, desired_length) -> list:
        current_length = self.query(f"SELECT COUNT(*) FROM {table_name}")[0][0]
        rows_to_remove = max(0, current_length - desired_length)
        
        if rows_to_remove > 0:
            rows_contents = self.query(f"SELECT * FROM {table_name} WHERE id <= {rows_to_remove}")
            return rows_contents
        else:
            return []


