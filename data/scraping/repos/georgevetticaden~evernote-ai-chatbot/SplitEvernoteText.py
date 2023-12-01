# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import datetime
import json
import time

from nifiapi.properties import PropertyDescriptor
from nifiapi.properties import StandardValidators
from nifiapi.properties import ExpressionLanguageScope
from nifiapi.flowfiletransform import FlowFileTransform, FlowFileTransformResult

from langchain.document_loaders import EverNoteLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class SplitEvernoteText(FlowFileTransform):
    class Java:
        implements = ['org.apache.nifi.python.processor.FlowFileTransform']
    class ProcessorDetails:
        dependencies = ['langchain', 'lxml', 'html2text']
        version = '0.0.1-SNAPSHOT'
        description = 'Converts a Evernote enex file into text documents that are split/chunked using Langchain text splitting utilities. The text are split to be optimized to create vector text embeddings'
        tags = ['text splitting', 'AI',  'evernote', "enex" 'langchain']


    def __init__(self, **kwargs):
        # Build Property Descriptors
        self.pdf_doc_url = PropertyDescriptor(
            name="PDF Document File",
            description="The full path to the PDF doc Url",
            required = True,
            expression_language_scope=ExpressionLanguageScope.FLOWFILE_ATTRIBUTES
        )
        self.chunk_size = PropertyDescriptor(
            name="Chunk Size",
            description="The number of characters that each text chunk should be",
            default_value = 1000,
            required = True,
        )
        self.chunk_overlap = PropertyDescriptor(
            name="Chunk Overlap",
            description="The number of characters to overlap between two contiguous text chunks",
            default_value = 100,
            required = True,
        )
        self.descriptors = [self.pdf_doc_url, self.chunk_size, self.chunk_overlap]

    def transform(self, context, flowFile):

        doc_url = context.getProperty(self.pdf_doc_url.name).evaluateAttributeExpressions(flowFile).getValue()

        self.logger.info("Inside transform of Chunking method for Evernote docs ")
        loader = EverNoteLoader(doc_url, load_single_document=False)
        doc = loader.load()

        chunk_size = context.getProperty(self.chunk_size).asInteger()
        chunk_overlap = context.getProperty(self.chunk_overlap).asInteger()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunked_docs = text_splitter.split_documents(doc)
        number_of_chunks = len(chunked_docs)

        self.logger.info("PDF Doc["+doc_url+"] was chunked into ["+str(number_of_chunks)+"] docs")

        # Convert he list of Langhcain Document into List of Json strings.
        # Also format the created/updated metadata to date ints so we can do metadata filtering in a vector db
        chunk_docs_json = []
        for doc in chunked_docs:

            # Assuming doc.metadata['created'] is a time.struct_time object. Convert to string
            created_str =  time.strftime("%Y%m%dT%H%M%SZ", doc.metadata['created'])
            updated_str = time.strftime("%Y%m%dT%H%M%SZ", doc.metadata['updated'])

            # Parse the date string to a datetime object
            create_dt = datetime.datetime.strptime( created_str, "%Y%m%dT%H%M%SZ")
            update_dt = datetime.datetime.strptime(updated_str , "%Y%m%dT%H%M%SZ")

            # Convert the datetime object to a Unix timestamp with only the date component
            # This is done so we can store this is a metadata time column in a vector db

            doc.metadata['created'] = int(create_dt.timestamp())
            doc.metadata['updated'] = int(update_dt.timestamp())

            doc_json = doc.json()
            chunk_docs_json.append(doc_json)

        # Convert the List of json strings into a single json string that can be return in the Flow File Result
        single_json_doc = json.dumps(chunk_docs_json)

        return FlowFileTransformResult(relationship="success", contents=single_json_doc)


    def getPropertyDescriptors(self):
        return self.descriptors
