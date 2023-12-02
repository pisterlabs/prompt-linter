import chromadb
from chromadb.config import Settings
import csv
from dotenv import load_dotenv, find_dotenv
import guidance
from langchain.embeddings import OpenAIEmbeddings
import os
import openai
import my_utils as U
from time import time as now
import sys
sys.path.append("/Users/allyne/Documents/GitHub/Unity-Agent/")

class MemoryManager:
    # gpt-3.5-turbo-1106
    # gpt-4-0613
    def __init__(self, model_name="gpt-3.5-turbo-1106", temperature=0, resume=False, retrieve_top_k=3, ckpt_dir="ckpt"):
        load_dotenv(find_dotenv())
        openai.api_key = os.getenv("OPENAI_API_KEY")

        guidance.llm = guidance.llms.OpenAI(model_name, temperature=temperature)
        self.llm=guidance.llm

        self.ckpt_dir = ckpt_dir
        self.retrieve_top_k = retrieve_top_k

        #TODO: May need to account for resume, or not. Not sure if need mkdir thingy too
        settings = Settings(chroma_db_impl="duckdb+parquet",
                                     persist_directory=f"../memory/{ckpt_dir}")
        print(f"Initializing memory in {settings.persist_directory}...")
        client = chromadb.Client(settings)
        client.persist()

        self.embeddings = OpenAIEmbeddings()
        self.client = client
        self.plansdb = client.get_or_create_collection(name="plansdb", embedding_function=self.embeddings)
        self.codedb = client.get_or_create_collection(name="codedb", embedding_function=self.embeddings)
    
    def _init_plan_memory(self, csv_path):
        t0=now()
        with open(csv_path, "r") as file:
            reader = csv.DictReader(file)
            for i, row in enumerate(reader):
                print(f"Embedding plan {i+1}...")
                user_query = row["User Query"]
                plan = row["Plan"]
                user_query_embedding = self.embeddings.embed_query(user_query)
                self.plansdb.add(
                    embeddings=[user_query_embedding],
                    metadatas=[{
                        "user_query": user_query,
                        "plan": plan,
                        }],
                    ids=[user_query]
                )
                U.dump_text(
                    f"User query:\n{user_query}\n\nPlan:\n{plan}", f"../memory/{self.ckpt_dir}/plans/{user_query}.txt"
                )
        return f"Intialized memory on planning in {now()-t0} seconds."
    
    def _init_code_memory(self, csv_path):
        t0=now()
        with open(csv_path, "r") as file:
            reader = csv.DictReader(file)
            for i, row in enumerate(reader):
                print(f"Embedding code {i+1}...")
                instruction = row["Instruction"]
                code = row["Code"]
                instruction_embedding = self.embeddings.embed_query(instruction)
                self.codedb.add(
                    embeddings=[instruction_embedding],
                    metadatas=[{
                        "instruction": instruction,
                        "code": code,
                        }],
                    ids=[instruction]
                )
                U.dump_text(
                    f"Instruction:\n{instruction}\n\nCode:\n{code}", f"../memory/{self.ckpt_dir}/code/{instruction}.txt"
                )
        return f"Intialized memory on coding in {now()-t0} seconds."
    
    def _get_code(self, instruction):
        instruction_embedding = self.embeddings.embed_query(instruction)
        # Retrieve 2 functions only
        k = min(self.codedb.count(), 5)
        if k==0:
            return []
        print(f"Retrieving {k} codes...")
        codes = self.codedb.query(
            query_embeddings=instruction_embedding,
			n_results=k,
            #where_document={"$contains":"search_string"}
			include=["metadatas"]
        )
        return codes["metadatas"][0][:2]

    def _get_plan(self, user_query):
        user_query_embedding = self.embeddings.embed_query(user_query)
        k = min(self.plansdb.count(), 5)
        if k==0:
            return []
        print(f"Retrieving {k} plans...")
        plans = self.plansdb.query(
            query_embeddings=user_query_embedding,
            n_results=k,
            include=["metadatas"]
        )
        # Just do 2
        return plans["metadatas"][0][:2]

    def _add_new_code(self, info):
        instruction = info["instruction"]
        code = info["code"]
        instruction_embedding = self.embeddings.embed_query(instruction)
        self.codedb.add(
            embeddings=[instruction_embedding],
            metadatas=[{
                "instruction": instruction,
                "code": code,
                }],
            ids=[instruction] #TODO: Account for repeated instructions
        )
        U.dump_text(
            f"Instruction:\n{instruction}\n\nCode:\n{code}", f"../memory/{self.ckpt_dir}/code/{instruction}.txt"
        )
        return f"Added code for instruction \"{instruction}\""
    
    def _add_new_plan(self, info):
        user_query = info["user_query"]
        plan = info["plan"]
        user_query_embedding = self.embeddings.embed_query(user_query)
        self.plansdb.add(
            embeddings=[user_query_embedding],
            metadatas=[{
                "user_query": user_query,
                "plan": plan,
                }],
            ids=[user_query] #TODO: Account for repeated user queries
        )
        U.dump_text(
            f"User query:\n{user_query}\n\nPlan:\n{plan}", f"../memory/{self.ckpt_dir}/plans/{user_query}.txt"
        )
        return f"Added plan for user query \"{user_query}\""
    
    def _add_new_experience(self, obj):
        task = obj.task
        plan_function_map = obj.new_plan_function_map
        plans = list(plan_function_map.keys())
        plan_str = ""
        for i, plan in enumerate(plans):
            plan_str += f"{i+1}. {plan}\n"
        new_plan_dict = {
            "user_query": task,
            "plan": plan_str,
        }
        self._add_new_plan(new_plan_dict)
        all_code_dicts = []
        for plan, function in plan_function_map.items():
            code_dict = {
                "instruction": plan,
                "code": function,
            }
            all_code_dicts.append(code_dict)
            self._add_new_code(code_dict)
        return new_plan_dict, all_code_dicts
    
    def _delete_plan_memory(self):
        self.client.delete_collection(name="plansdb")
        return "Deleted plan memory."
    
    def _delete_code_memory(self):
        self.client.delete_collection(name="codedb")
        return "Deleted code memory."
    
    def _delete_one_plan(self, user_query):
        self.plansdb.delete(ids=[user_query])
        return f"Deleted plan for user query \"{user_query}\""
    
    def _delete_one_code(self, instruction):
        self.codedb.delete(ids=[instruction])
        return f"Deleted code for instruction \"{instruction}\""