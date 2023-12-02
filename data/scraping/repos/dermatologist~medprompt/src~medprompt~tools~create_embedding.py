"""
 Copyright 2023 Bell Eapen

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""


import os
from typing import Optional, Type

from fhir.resources.bundle import Bundle
from langchain.callbacks.manager import (AsyncCallbackManagerForToolRun,
                                         CallbackManagerForToolRun)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.tools import BaseTool
from langchain.vectorstores import Redis
from pydantic import BaseModel, Field

from .. import MedPrompter, get_time_diff_from_today


class BundleInput(BaseModel):
    bundle_input: Bundle = Field()

class CreateEmbeddingFromFhirBundle(BaseTool):
    """
    Creates an embedding from a FHIR Bundle resource.
    """
    name = "Create Embedding From FHIR Bundle"
    description = """
    Creates an embedding from a FHIR Bundle resource.
    """
    args_schema: Type[BaseModel] = BundleInput

    # Embedding model
    EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # Redis Connection Information
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

    # Create vectorstore
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # Index schema
    current_file_path = os.path.abspath(__file__)
    parent_dir = os.path.dirname(current_file_path)
    schema_path = os.path.join(parent_dir, "schema.yml")
    INDEX_SCHEMA = schema_path

    def _run(
            self,
            bundle_input: Bundle = None,
            run_manager: Optional[CallbackManagerForToolRun] = None
            ) -> Bundle:
        prompt = MedPrompter()
        chunks = []
        patient_id = "random"
        for entry in bundle_input.entry:
            resource = entry.resource
            if resource.resource_type == "Patient":
                patient_id = resource.id
            if resource.resource_type == "Patient" or resource.resource_type == "Observation" \
                or resource.resource_type == "Condition" or resource.resource_type == "Procedure" \
                or resource.resource_type == "MedicationRequest" or resource.resource_type == "DiagnosticReport":
                obj: dict = resource.dict()
                obj["time_diff"] = get_time_diff_from_today
                chunk = {
                    "page_content": prompt.generate_prompt(obj).replace("\n", " "),
                    "metadata": resource.id
                }
                chunks.append(chunk)
        # Store in Redis
        _ = Redis.from_texts(
            # appending this little bit can sometimes help with semantic retrieval
            # especially with multiple companies
            # texts=[f"Company: {company_name}. " + chunk.page_content for chunk in chunks],
            texts=[chunk.page_content for chunk in chunks],
            metadatas=[chunk.metadata for chunk in chunks],
            embedding=self.embedder,
            index_name=patient_id,
            index_schema=self.INDEX_SCHEMA,
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379")
        )
        return chunks
    async def _arun(
            self,
            bundle_input: Bundle = None,
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None
            ) -> Bundle:
        raise NotImplementedError("Async not implemented yet")