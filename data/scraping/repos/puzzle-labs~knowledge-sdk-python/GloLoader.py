# import supporting packages and modules
from abc import ABC
import yaml
import os
import tempfile
import requests
from urllib.parse import urlparse
from typing import List
import json
import re

# import langchain modules
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders import OnlinePDFLoader
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.document_loaders import WebBaseLoader

# import puzzle-knowledge-sdk modules
from .Ngram import AddAlphaSmooth

class GloLoader(BaseLoader, ABC):
    """Loader class for `.glo` files.
    
    Defaults to check for local file, but if file is a web path, it will download it to a temporary file, use it, then clean up the temporary file after completion.
    """
    
    def __init__(self, file_path: str):
        """Initializes the loader with the file path."""
        self.file_path, self.web_path = self._process_file_path(file_path)
        
    def _process_file_path(self, file_path: str):
        """Handles file checking, URL validity checking, and downloading if necessary."""
        web_path = None  # Initialize web_path locally

        if "~" in file_path:
            file_path = os.path.expanduser(file_path)

        if os.path.isfile(file_path):
            return file_path, web_path  # Return a tuple with two values
        elif self._is_valid_url(file_path):
            temp_dir = tempfile.TemporaryDirectory()
            self.temp_dir = temp_dir
            _, suffix = os.path.splitext(file_path)
            temp_glo = os.path.join(temp_dir.name, f"tmp{suffix}")

            if self._is_s3_url(file_path):
                web_path = file_path
            else:
                r = requests.get(file_path)
                
                if r.status_code != 200:
                    print(file_path)

                web_path = file_path
                with open(temp_glo, mode="wb") as f:
                    f.write(r.content)
                    
            return str(temp_glo), web_path  # Return a tuple with two values
        else:
            raise ValueError("File path %s is not a valid file or URL" % file_path)


    def __del__(self) -> None:
        if hasattr(self, "temp_dir"):
            self.temp_dir.cleanup()

    @staticmethod
    def _is_valid_url(url: str) -> bool:
        """Check if the url is valid."""
        parsed = urlparse(url)
        return bool(parsed.netloc) and bool(parsed.scheme)

    @staticmethod
    def _is_s3_url(url: str) -> bool:
        """check if the url is S3"""
        try:
            result = urlparse(url)
            if result.scheme == "s3" and result.netloc:
                return True
            return False
        except ValueError:
            return False
        
    @staticmethod
    def _is_html_page(url):
        try:
            response = requests.head(url)
            content_type = response.headers.get('content-type', '').lower()
            return 'text/html' in content_type
        except requests.exceptions.RequestException:
            return False

    @property
    def source(self) -> str:
        return self.web_path if self.web_path is not None else self.file_path
    
    def import_data(self) -> dict:
        """Load concept documents."""
        if isinstance(self.file_path, str):
            loaded = False
            
            # load data json or yaml or glo
            # yaml load
            with open(self.file_path, 'r') as glo_file:
                glo_text = glo_file.read()
            try:
                data = yaml.safe_load(glo_text)
                return data
            except:
                pass
            # json load
            try:
                with open(self.file_path, 'r') as json_file:
                    data = json.load(json_file)
                return data
            except:
                pass
                
            if not loaded:
                raise ValueError("Error parsing file: Not in valid JSON or YAML formats")
            
        elif isinstance(self.file_path, dict):
            data = self.file_path
            return data
        
    @staticmethod
    def load_link(link, type, concept_name, load) -> Document:
        if type == 'uri':
            if link.endswith(".pdf"):
                if load:
                    loader = OnlinePDFLoader(link)
                    data_load = loader.load()[0]
                    data_load.page_content = re.sub(r'\n+', '\n', data_load.page_content)
                else:
                    data_load = Document(page_content="")
                data_load.metadata = {
                    "concept": concept_name,
                    "type": "link-pdf",
                    "link_type": "uri",
                    "source": link,
                    "load_status": load
                }
                return data_load
            else:
                try:
                    if load:
                        loader = WebBaseLoader(link)
                        data_load = loader.load()[0]
                        data_load.page_content = re.sub(r'\n+', '\n', data_load.page_content)
                    else:
                        data_load = Document(page_content="")
                    data_load.metadata = {
                        "concept": concept_name,
                        "type": "link-html",
                        "link_type": "uri",
                        "source": link,
                        "load_status": load
                    }
                    return data_load
                except Exception as e:
                    raise ValueError(f"Error loading link {link}: {e}")
        elif type == 'text':
            data_load = Document(
                page_content=link if load else "",
                metadata={
                    "concept": concept_name,
                    "type": "link-text",
                    "link_type": "text",
                    "source": "text",
                    "load_status": load
                }
            )
            return data_load
        elif type == 'glo':
            loader = GloLoader(link)
            data_load = loader.load()
            text = GloLoader.transform(documents=data_load, include_task=False)
            data_load = Document(
                page_content=text if load else "",
                metadata={
                    "concept": concept_name,
                    "type": "link-glo",
                    "link_type": "glo",
                    "source": link if isinstance(link, str) else f"Glossary: {link['name']}",
                    "load_status": load
                }
            )
            return data_load
        else:
            raise ValueError(f"Invalid link given: Can only process html, pdf, text, and glo.")
    
    def load(self, loadLinks=False) -> List[Document]:
        data = self.import_data()
        
        documents = []
        
        if "concepts" in data:
            concepts = data["concepts"]
            for concept in concepts:
                if "name" in concept and "explanation" in concept:
                    content = f"NAME: {concept['name']}\nEXPLANATION: {concept['explanation']}"
                    new_document = Document(
                        page_content=re.sub(r'\n+', '\n', content), 
                        metadata={
                            "glo_name": data.get("name", ""),
                            "topic": data.get("topic", ""), 
                            "audience": data.get("audience", ""), 
                            "concept": concept.get("name", ""), 
                            "type": "content", 
                            "source": self.file_path if isinstance(self.file_path, str) else self.file_path["name"],
                            "links": []
                            }
                        )
                    
                    if "links" in concept.keys():
                        for link in concept["links"]:
                            if "uri" in link.keys():
                                new_document.metadata["links"].append(GloLoader.load_link(link["uri"], type="uri", concept_name=concept["name"], load=loadLinks))
                            elif "text" in link.keys():
                                new_document.metadata["links"].append(GloLoader.load_link(link["text"], type="text", concept_name=concept["name"], load=loadLinks))
                            elif "glo" in link.keys():
                                new_document.metadata["links"].append(GloLoader.load_link(link["glo"], type="glo", concept_name=concept["name"], load=loadLinks))
                                
                    documents.append(new_document)
                    
                else:
                    raise ValueError("Concepts must have a name and explanation")
                    
        return documents
    
    @staticmethod
    def calculate_score(sample, source, n=2, scope="word"):
        # to lower
        sample = sample.lower()
        source = source.lower()
        
        n = 2
        scope = "word"
        
        try:
            if n in [1, 2, 3]:
                if scope == "word":
                    # preprocess pattern
                    pattern = r'([!@#$%^&*()_+{}\[\]:;"\'<>,.?/\|\\])'
                    prep_source = [re.sub(pattern, r' \1', source).split(" ")]
                    prep_sample = [re.sub(pattern, r' \1', sample).split(" ")]
                elif scope == "char":
                    prep_source = [list(source)]
                    prep_sample = [list(sample)]
                
                dist_source = AddAlphaSmooth(n, prep_source)
                
                score = dist_source.getSentenceLogLikelihood(prep_sample[0])
                return score
            else:
                raise ValueError(f"ngram methods must have n in [1, 2, 3]")
        except ValueError as e:
            raise ValueError(f"Concept ranking failed: {e}")
    
    @staticmethod
    def transform(query: str, documents: List[Document], header: str=None, task: str=None, rank_function=None, additional_args: dict={}):
        if header is None or header == "":
            # glo name
            glo_name = ""
            for doc in documents:
                if isinstance(doc, Document):
                    glo_name = doc.metadata.get("glo_name", "")
                    if glo_name != "":
                        break
            header = f"GLOSSARY: {glo_name}"
            
        if task is None or task == "":
            task = "TASK: This is a glossary of concepts for your reference throughout this conversation. You should prioritize this information when answering any questions asked by the user."
            
        max_width = additional_args.get("max_width", 1024)
        
        if rank_function is None:
            def fn(documents, max_width):
                context = "CONCEPTS: \n"
                
                # collect concepts
                concepts = []
                for doc in documents:
                    if isinstance(doc, Document):
                        if doc.metadata["type"] == "content":
                            concepts.append(doc.page_content)
                            
                # select maximum possible number of concepts to fit in context window
                filtered_concepts = []
                max_width -= len(context)
                for concept in concepts:
                    if len(concept) < max_width:
                        filtered_concepts.append(concept)
                        max_width -= len(concept) - 1
                    else:
                        break
            
                # format concepts
                concepts = "\n".join([concept for concept in filtered_concepts])
                context += concepts
                
                return context
                            
            rank_function = fn
            
        template = "{header}\n{concepts}\n\n{task}"
        max_width -= len(template.format(header="", concepts="", task="")) + len(header) + len(task)
        parameters = {
            "query": query,
            "documents": documents,
            "max_width": max_width
        }
        if additional_args is not {} and additional_args is not None:
            parameters.update(additional_args)
        
        concepts = rank_function(**{k: v for k, v in parameters.items() if k in rank_function.__code__.co_varnames})
            
        prompt_template = PromptTemplate(
            input_variables = ["header", "concepts", "task"],
            template = template
        )
        
        prompt = prompt_template.format(header=header, concepts=concepts, task=task)
        
        return prompt
    
    
    # Concepts search and transform
    @staticmethod
    def rank_by_concepts(query, documents, max_width=1024):
        # context variable
        context = "CONCEPTS: \n"
        
        # collect concepts
        concepts = []
        for doc in documents:
            if isinstance(doc, Document):
                if doc.metadata["type"] == "content":
                    concepts.append(doc.page_content)
        
        # compute score of concepts
        scores = []
        for concept in concepts:
            score = GloLoader.calculate_score(query, concept, n=1, scope="word")
            scores.append(
                {
                    "concept": concept,
                    "score": score
                }
            )
                    
        # sort concepts by score
        sorted_concepts = sorted(scores, key=lambda x: x["score"], reverse=True)
        
        # select maximum possible number of concepts to fit in context window
        filtered_concepts = []
        max_width -= len(context)
        for concept in sorted_concepts:
            if len(concept["concept"]) < max_width:
                filtered_concepts.append(concept)
                max_width -= len(concept["concept"]) - 1
            else:
                break
            
        # format concepts
        concepts = "\n".join([concept["concept"] for concept in filtered_concepts])
        context += concepts
        
        return context
    
    
    # links search and transform
    @staticmethod
    def rank_by_links(query, documents, max_width=1024, chunk_size=512, chunk_overlap=128):
        # context variable 
        context = "CONCEPTS: \n"
        
        # text splitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        link_chunks = []
        
        # split texts
        for doc in documents:
            if isinstance(doc, Document):
                link_documents = doc.metadata.get("links", [])
                for link_doc in link_documents:
                    if link_doc.metadata["type"] in ["link-html", "link-pdf", "link-text"]:
                        if not link_doc.metadata["load_status"]:
                            link_doc = GloLoader.load_link(
                                link=link_doc.metadata["source"], 
                                type=link_doc.metadata["link_type"], 
                                concept_name=doc.metadata["concept"], 
                                load=True
                            )
                        link_chunks.extend(text_splitter.split_documents([link_doc]))

        # build vector store
        if len(link_chunks) == 0:
            similar_texts = []
        else:
            links_vectorstore = FAISS.from_documents(link_chunks, OpenAIEmbeddings())
            relevant_chunks = links_vectorstore.similarity_search(query)
            similar_texts = [doc.page_content for doc in relevant_chunks]
            
        # select maximum possible number of chunks to fit in context window
        filtered_chunks = []
        max_width -= len(context)
        for chunk in similar_texts:
            if len(chunk) < max_width:
                filtered_chunks.append(chunk)
                max_width -= len(chunk) - 1
            else:
                break
            
        # format chunks
        chunks = "\n".join([chunk for chunk in filtered_chunks])
        context += chunks
        
        return context
    
    
    # concepts+links search and transform
    @staticmethod
    def rank_by_concepts_and_links(query, documents, max_width=1024, chunk_size=512, chunk_overlap=128):
        # context variable
        context = "CONCEPTS: \n"
        
        # collect concepts
        concepts = []
        for doc in documents:
            if isinstance(doc, Document):
                if doc.metadata["type"] == "content":
                    concepts.append(doc)
        
        # compute score of concepts
        scores = []
        for concept in concepts:
            score = GloLoader.calculate_score(query, concept.page_content, n=1, scope="word")
            scores.append(
                {
                    "concept": concept,
                    "score": score
                }
            )
                    
        # sort concepts by score
        sorted_concepts = sorted(scores, key=lambda x: x["score"], reverse=True)
        
        # text splitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        link_chunks = []
        
        # select maximum possible number of chunks to fit in context window
        filtered_concepts = []
        max_width -= len(context)
        for concept in sorted_concepts:
            if len(concept["concept"].page_content) < max_width:
                # link qa here
                link_documents = concept["concept"].metadata.get("links", [])
                for link_doc in link_documents:
                    if not link_doc.metadata["load_status"]:
                        link_doc = GloLoader.load_link(
                            link=link_doc.metadata["source"], 
                            type=link_doc.metadata["link_type"], 
                            concept_name=concept["concept"].metadata["concept"], 
                            load=True
                        )
                        
                    if link_doc.metadata["type"] in ["link-html", "link-pdf", "link-text"]:
                        link_chunks.extend(text_splitter.split_documents([link_doc]))

                # build vector store
                if len(link_chunks) == 0:
                    similar_texts = []
                else:
                    links_vectorstore = FAISS.from_documents(link_chunks, OpenAIEmbeddings())
                    relevant_chunks = links_vectorstore.similarity_search(query)
                    similar_texts = [doc.page_content for doc in relevant_chunks]
                
                filtered_concepts.append(concept["concept"].page_content + "\nEXCERPTS: \n" + "\n".join(similar_texts[:3]))
                max_width -= len(filtered_concepts[-1])
            else:
                break
            
        # format chunks
        concepts = "\n".join([concept for concept in filtered_concepts])
        context += concepts
        
        return context