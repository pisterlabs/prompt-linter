import time
import json
import os
import logging
import traceback
import cachetools.func
import hashlib
import uuid

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field
from qdrant_client.http import models as rest
from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
from qdrant_retriever import QDrantVectorStoreRetriever
from langchain.retrievers import ContextualCompressionRetriever
from cohere_rerank import CohereRerank
from langchain.schema import Document
from datetime import datetime, timedelta
from qdrant_client.http.models import PayloadSchemaType
from functions_and_agents_metadata import AuthAgent
from typing import List
from concurrent.futures import ThreadPoolExecutor

class DiscoverGroupsModel(BaseModel):
    query: str
    auth: AuthAgent

class DiscoverGroupsManager:

    def __init__(self, rate_limiter, rate_limiter_sync):
        load_dotenv()  # Load environment variables
        self.QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
        os.getenv("COHERE_API_KEY")
        self.QDRANT_URL = os.getenv("QDRANT_URL")
        self.index = None
        self.rate_limiter = rate_limiter
        self.rate_limiter_sync = rate_limiter_sync
        self.max_length_allowed = 1024
        self.collection_name = "discover_groups"
        self.client = QdrantClient(url=self.QDRANT_URL, api_key=self.QDRANT_API_KEY)
        self.inited = False
        
        
    def create_new_groups_retriever(self, api_key: str):
        """Create a new vector store retriever unique to the agent."""
        # create collection if it doesn't exist (if it exists it will fall into finally)
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=rest.VectorParams(
                    size=1536,
                    distance=rest.Distance.COSINE,
                ),
            )
            self.client.create_payload_index(self.collection_name, "metadata.namespace_id", field_schema=PayloadSchemaType.KEYWORD)
        except:
            logging.info(f"DiscoverGroupsManager: loaded from cloud...")
        finally:
            logging.info(
                f"DiscoverGroupsManager: Creating memory store with collection {self.collection_name}")
            vectorstore = Qdrant(self.client, self.collection_name, OpenAIEmbeddings(openai_api_key=api_key))
            compressor = CohereRerank(top_n=5)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=QDrantVectorStoreRetriever(
                    rate_limiter=self.rate_limiter, rate_limiter_sync=self.rate_limiter_sync, collection_name=self.collection_name, client=self.client, vectorstore=vectorstore,
                )
            )
            return compression_retriever

    def generate_id_from_name(self, name):
        hash_object = hashlib.sha256(name.encode())
        # Use hexdigest for a hexadecimal string representation
        return str(uuid.UUID(bytes=hash_object.digest()[:16]))

    async def transform(self, namespace_id, data):
        now = datetime.now().timestamp()
        result = []
        # Continue with your existing logic but using `items_to_process`
        for item in data:
            page_content = {'name': item['name'], 'description': str(item['description'])[:960]}
            lenData = len(str(page_content))
            if lenData > self.max_length_allowed:
                logging.info(
                    f"DiscoverGroupsManager: transform tried to create an agent that surpasses the maximum length allowed max_length_allowed: {self.max_length_allowed} vs length of data: {lenData}")
                continue
            metadata = {
                "id": self.generate_id_from_name(item['name']),
                "namespace_id": namespace_id,
                "last_accessed_at": now,
            }
            doc = Document(
                page_content=json.dumps(page_content),
                metadata=metadata
            )
            result.append(doc)
        return result

    def extract_name(self, documents):
        result = []
        seen = set()  # Track seen combinations of name
        for doc in documents:
            # Parse the page_content string into a Python dict
            text = json.loads(doc.page_content)
            name = text.get('name')

            # Check if this combination has been seen before
            if name not in seen:
                result.append({'name': name})
                seen.add(name)

        return result
    
    async def pull_groups(self, agent_input: DiscoverGroupsModel):
        """Fetch groups based on a query."""
        if self.inited is False:
            try:
                self.client.get_collection(self.collection_name)
            except Exception as e:
                logging.warning(f"DiscoverGroupsManager: pull_groups exception {e}\n{traceback.format_exc()}")
                self.inited = True
        memory = self.load(agent_input.auth.api_key)
        response = []
        #loop = asyncio.get_event_loop()
        try:
            documents = await self.get_retrieved_nodes(memory,
                agent_input.query, agent_input.auth.namespace_id)
            if len(documents) > 0:
                parsed_response = self.extract_name(documents)
                response.append(parsed_response)
                # update last_accessed_at
                ids = [doc.metadata["id"] for doc in documents]
                for doc in documents:
                    doc.metadata.pop('relevance_score', None)
                await self.rate_limiter.execute(memory.base_retriever.vectorstore.aadd_documents, documents, ids=ids)
                #loop.run_in_executor(None, self.prune_groups)
        except Exception as e:
            logging.warning(f"DiscoverGroupsManager: pull_groups exception {e}\n{traceback.format_exc()}")
        finally:
            return response

    async def get_retrieved_nodes(self, memory: ContextualCompressionRetriever, query_str: str, namespace_id: str):
        kwargs = {}
        # if user provided then look for null or direct matches, otherwise look for null so it matches public groups
        if namespace_id != "":
            filter = rest.Filter(
                should=[
                    rest.FieldCondition(
                        key="metadata.namespace_id",
                        match=rest.MatchValue(value=namespace_id),
                    ),
                    rest.IsNullCondition(
                        is_null=rest.PayloadField(key="metadata.namespace_id")
                    )
                ]
            )
            kwargs["user_filter"] = filter
        else:
            filter = rest.Filter(
                should=[
                    rest.IsNullCondition(
                        is_null=rest.PayloadField(key="metadata.namespace_id")
                    )
                ]
            )
            kwargs["user_filter"] = filter
        return await memory.aget_relevant_documents(query_str, **kwargs)

    def get_document_by_name(self, memory: ContextualCompressionRetriever, name: str) -> Document:
        return memory.base_retriever.get_key_value_document("metadata.name", name)

    @cachetools.func.ttl_cache(maxsize=16384, ttl=36000)
    def load(self, api_key: str):
        """Load existing index data from the filesystem for a specific user."""
        start = time.time()
        memory = self.create_new_groups_retriever(api_key)
        end = time.time()
        logging.info(
            f"DiscoverGroupsManager: Load operation took {end - start} seconds")
        return memory

    async def push_groups(self, auth: AuthAgent, groups):
        """Update the current index with new groups."""
        memory = self.load(auth.api_key)
        try:
            logging.info("DiscoverGroupsManager: pushing groups...")
            all_docs = []
            transformed_groups = await self.transform(
                        auth.namespace_id, groups)
            all_docs.extend(transformed_groups)
            ids = [doc.metadata["id"] for doc in all_docs]
            await self.rate_limiter.execute(memory.base_retriever.vectorstore.aadd_documents, all_docs, ids=ids)
        except Exception as e:
            logging.warning(f"DiscoverGroupsManager: push_groups exception {e}\n{traceback.format_exc()}")
        finally:
            return "success"

    def prune_groups(self):
        """Prune groups that haven't been used for atleast six weeks."""
        def attempt_prune():
            current_time = datetime.now()
            six_weeks_ago = current_time - timedelta(weeks=6)
            filter = rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="metadata.last_accessed_at", 
                        range=rest.Range(lte=six_weeks_ago.timestamp()), 
                    )
                ]
            )
            self.client.delete(collection_name=self.collection_name, points_selector=filter)
        try:
            attempt_prune()
        except Exception as e:
            logging.warning(f"DiscoverGroupsManager: prune_groups exception {e}\n{traceback.format_exc()}")
            # Attempt a second prune after reload
            try:
                attempt_prune()
            except Exception as e:
                # If prune after reload fails, propagate the error upwards
                logging.error(f"DiscoverGroupsManager: prune_groups failed after reload, exception {e}\n{traceback.format_exc()}")
                raise
        return True

    def delete_groups(self, auth: AuthAgent, groups: List[str]):
        """Delete groups from the Qdrant collection."""
        try:
            logging.info("DiscoverGroupsManager: deleting groups...")
            filter_conditions = rest.Filter(
                should=[
                    rest.FieldCondition(
                        key="metadata.namespace_id",
                        match=rest.MatchValue(value=auth.namespace_id),
                    ),
                    rest.FieldCondition(
                        key="name",
                        match=rest.MatchAny(any=groups),
                    )
                ]
            )
            self.client.delete(collection_name=self.collection_name, points_selector=filter_conditions)
            return "success"
        except Exception as e:
            logging.warning(f"DiscoverGroupsManager: delete_groups exception {e}\n{traceback.format_exc()}")
            return str(e)

