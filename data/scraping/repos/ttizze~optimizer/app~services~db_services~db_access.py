from typing import Any, Dict, List, Type, Union

from pydantic import Field, BaseModel

from supabase import create_client, Client

from langchain.embeddings import OpenAIEmbeddings
import os
import openai
from datetime import datetime
import requests
import httpx
import pytz
import traceback
import requests
import json
import csv
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
supabaseurl = os.getenv("SUPABASE_URL")
supabasekey = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabaseurl, supabasekey)


class DBLayer(BaseModel):
    def save_to_table_with_embed(
        self, user_id: int, text: str, table: str, **columns
    ) -> httpx.Response:
        """Save message to database."""
        # embed

        embedding_model = OpenAIEmbeddings()
        query_embedding = embedding_model.embed_query(text)
        data = {
            "user_id": user_id,
            "content": text,
            "embed": query_embedding,
        }

        for key, value in columns.items():
            data[key] = value

        result = supabase.table(table).insert(data).execute()
        return result

    def save_csv_to_database_with_embed(self, csv_file: str, table: str):
        """Read CSV file and save each row to the database with embeddings."""

        embedding_model = OpenAIEmbeddings()
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                query_embedding = embedding_model.embed_query(row["generatedSummary"])
                data = {
                    "content":row["generatedSummary"],
                    "embed" :embedding_model.embed_query(query_embedding)
                }
                result = supabase.table(table).insert(data).execute()
        return result
    # line_user_idをuser_idに変換
    def get_user_id_from_line_id(self, line_user_id: str) -> int:
        user_id = (
            supabase.table("users")
            .select("id")
            .eq("line_user_id", line_user_id)
            .execute()
        )
        if user_id.data:
            user_id = user_id.data[0]["id"]
            return user_id
        else:
            return None

    def create_user(self, line_user_id: str) -> int:
        user_id = (
            supabase.table("users")
            .insert({"line_user_id": line_user_id})
            .execute()
        )
        return user_id.data[0]["id"]

    def search_support(self,profile:str) -> str:
        embedding_model = OpenAIEmbeddings()
        query_embedding = embedding_model.embed_query(profile)
        result = supabase.rpc("match_documents", {"query_embedding": query_embedding,"match_count":5}).execute()
        supports =""
        if result.data:
            #リストを連結して一つの文字列にする
            for i in range(len(result.data)):
                supports += "[ "+ result.data[i]["title"] + " ]\n" + result.data[i]["content"] + "\n\n"
            return supports
        else:
            return None

    def get_profile(self,user_id:int) -> str:
        profile = (
            supabase.table("users")
            .select("profile")
            .eq("id", user_id)
            .execute()
        )
        if profile.data:
            profile = profile.data[0]["profile"]
            return profile
        else:
            return None
        
    def update_profile(self,user_id:int,profile:str) -> str:
        profile = (
            supabase.table("users")
            .update({"profile": profile})
            .eq("id", user_id)
            .execute()
        )
        return profile.data[0]["profile"]