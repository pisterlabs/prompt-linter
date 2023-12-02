import os
import openai
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

# import chromadb
# from chromadb.utils import embedding_functions

from underthesea import word_tokenize
from qdrant_client import QdrantClient

from anthropic import Anthropic

import vertexai
from google.cloud import translate
from vertexai.preview.language_models import TextGenerationModel

from langchain.vectorstores import Qdrant
from langchain.embeddings import CohereEmbeddings


# Load environment variables from .env file
load_dotenv()

class BookQA:
    def __init__(self, 
                 vector_path:str=None, 
                 user:str=None,
                 collection_name:str=None,
                 query:str=None, #Put some questions / queries here
                 llm:str='chatgpt', #Or 'palm2' # or 'claude'
                 vmethod:str='qdrant',
                 book_lang:str='en',
                 top_k_searching:int=5,
                 ):
        
        self.vector_path = vector_path
        self.fact_path = f'database/{user}/facts_db/facts_vector_db'
        self.fact_json = os.getcwd() + '/' + f'database/{user}/facts_db/txt_db/qadb.json'
        self.collection_name = collection_name
        self.vmethod = vmethod
        self.llm = llm
        self.book_lang = book_lang
        self.top_k_searching = top_k_searching
        
        self.ANTHROPIC_API_KEY = os.environ['ANTHROPIC_API_KEY']
        
        self.qdrant_url = os.environ['QDRANT_URL']
        self.qdrant_api_key = os.environ['QDRANT_API_KEY']
        self.qdrant_client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
        
        openai.api_key = os.environ['OPENAI_API_KEY']
        
        self.embeddings = CohereEmbeddings(model="multilingual-22-12", 
                                           cohere_api_key=os.environ['COHERE_API_KEY'])
        
        self.vdatabase = Qdrant(client=self.qdrant_client, 
                                collection_name=self.collection_name, 
                                embeddings=self.embeddings)
        
    def bookQnA(self, question):
        self.query = question
        self.searching()
        try:
            self.matched_ans, self.matching_score = self.facts_matching()
        except:
            st.warning('Could not found Facts Collection!')

        if self.matching_score > 0.95: 
            llm_answer, response_time, refers = self.matched_ans, 0, ['faq', 'FAQ']
        else:
            self.matched_ans = ''
            self.prompting()
            if self.llm == 'palm2':
                # print('Google Responding...\n')
                try:
                    llm_answer, response_time = self.responding_google()
                except:
                    pass
            elif self.llm == 'openai':
                # print('OpenAI Responding...\n')
                llm_answer, response_time = self.responding_openai()
            elif self.llm == 'claude':
                # print('Claude Responding...\n')
                llm_answer, response_time = self.responding_claude()

            refers = self.references()

        return llm_answer, response_time, refers
        
    #DATABASE-SEARCHING----------------------------------------------------------
    def searching(self):
        if self.book_lang == 'vi':
            _query = word_tokenize(self.query, format="text")
        else:
            _query = self.query
                    
        self.search_results = self.vdatabase.similarity_search_with_score(_query, k=self.top_k_searching)
        print(f'Finished Search\nTop k = {len(self.search_results)}\n')
        return self.search_results

    def references(self):
        #Return the references
        print("\n")
        try:
            for i in range(len(self.search_results)):
                _pdf_name = self.search_results[i][0].metadata["source"].split("/")[-1]
                _ref_page = self.search_results[i][0].metadata["page"]
                print(f"Reference {i+1}: Page {_ref_page} in the file {_pdf_name}")
        except KeyError:
            # print("No References from this document\n")
            pass
        return self.search_results
    #FACTS MATCHING-------------------------------------------------------------------
    def facts_matching(self):
        collection_name = st.session_state.user + '_factsdb'
        import cohere
        cohere_client = cohere.Client(api_key="4ECOTqDXJpIYhxMQhUZxY12PPSqvgtYFclJm4Gnz")
        
        results = self.qdrant_client.search(collection_name=collection_name,
                    query_vector=cohere_client.embed(texts=[self.query],
                                                    model='multilingual-22-12',
                                                    ).embeddings[0],
                    limit=1
                    )
        
        #Return
        matched_ans = results[0].payload['metadata']['answer']
        matching_score = results[0].score

        return matched_ans, matching_score
    
    #PROMPTING-------------------------------------------------------------------
    def prompting(self):
        _search_info = " --- " + " --- ".join([self.search_results[i][0].page_content 
                                      for i in range(len(self.search_results))]) + " --- "
        #Translate to EN if using Palm2-----------------------------------
        if self.llm == 'palm2' and self.book_lang == 'vi':
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'assets/credential/ambient-hulling-389607-89c372b7af63.json'
            self.PARENT = f"projects/{'ambient-hulling-389607'}"
            #---VI-EN_TRANSLATION----------------------------------------
            _search_info = _search_info.replace("_"," ")
            translated_search_info = self.translate_text(_search_info, target_language_code='en')
            _search_info = translated_search_info.translated_text
        self.prompt = f"""
        You will be provided with the question which is delimited by XML tags and the \
        context delimited by triple backticks. 
        The context contains some long paragraphs and 1 reference which delimited by triple dash. \
        <tag>{self.query}</tag>
        ````\n{_search_info}```\n{self.matched_ans}\n```
        """
        
        if self.book_lang=='vi':
            self.prompt = self.prompt.replace("_"," ")
                    
    def tiktoken_len(self, text):
        import tiktoken
        tokenizer = tiktoken.get_encoding('cl100k_base')
        tokens = tokenizer.encode(
            text,
            disallowed_special=()
        )
        return len(tokens)
    
    #CHATPGT-RESPONSES-----------------------------------------------------------
    def responding_openai(self):
        prompt_token_length = self.tiktoken_len(self.prompt)
        if prompt_token_length > 16000:
            print("Length of Prompt exceeds the limitation of LLM input. Task closed!")
        else:    
            _start = datetime.now()
            if self.book_lang == 'vi':
                _sys_messages = [{"role": "system", "content": "You are a helpful assistant that gives a comprehensive answer  \
                                in Vietnamese from the given information"},
                                {"role": "user", "content": self.prompt}]
            else:
                _sys_messages = [{"role": "system", "content": "You are a helpful assistant that gives a comprehensive answer  \
                                from the given information"},
                                {"role": "user", "content": self.prompt}]
            response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-16k",
                    messages=_sys_messages,
                    max_tokens = 16000 - prompt_token_length, #Maximum length of tokens is 4096 included Prompt Tokens
                    n=1,
                    temperature=0.1,
                    top_p=0.7,
                )
            self.results = response.choices[0].message.content
            self.chatgpt_tokens = response.usage.total_tokens
            #Response Time (s)
            self.chatgpt_response_time = (datetime.now() - _start)
            
        return self.results, self.chatgpt_response_time
        
    
    #GOOGLE-RESPONSES------------------------------------------------------------
    def responding_google(self):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'assets/credential/ambient-hulling-389607-89c372b7af63.json'
        self.PARENT = f"projects/{'ambient-hulling-389607'}"
        #---VI-EN_TRANSLATION----------------------------------------
        # self.prompt = self.translate_text(self.prompt, target_language_code='en')
        # print(f"{self.prompt.translated_text}")

        #---PALM2-RESPONSE-------------------------------------------
        en_response = self.palm_response(self.prompt)
        # print(en_response)
        if self.book_lang == 'vi':
            #---EN-VI_TRANSLATION----------------------------------------
            vi_response = self.translate_text(en_response, target_language_code='vi')
            return vi_response.translated_text, self.palm2_response_time
        else:
            return en_response, self.palm2_response_time
        #---TRANSLATION----------------------------------------------
    def translate_text(self, text: str, target_language_code: str) -> translate.Translation:
        client = translate.TranslationServiceClient()

        response = client.translate_text(
            parent=self.PARENT,
            contents=[text],
            target_language_code=target_language_code,
            )
        return response.translations[0]

    #---PALM2-RESPONSE-----------------------------------------------
    def predict_large_language_model_gg(self,
                                            project_id: str,
                                            model_name: str,
                                            temperature: float,
                                            max_decode_steps: int,
                                            top_p: float,
                                            top_k: int,
                                            content: str,
                                            location: str = "us-central1",
                                            tuned_model_name: str = "",
                                            ):
        """Predict using a Large Language Model."""
        vertexai.init(project=project_id, location=location)

        model = TextGenerationModel.from_pretrained(model_name)

        if tuned_model_name:
            model = model.get_tuned_model(tuned_model_name)

        response = model.predict(
                                content,
                                temperature=temperature,
                                max_output_tokens=max_decode_steps,
                                top_k=top_k,
                                top_p=top_p,)
        return response.text

    def palm_response(self, content):
        _start = datetime.now()
        res = self.predict_large_language_model_gg(project_id="ambient-hulling-389607", 
                                                    model_name="text-bison@001", 
                                                    temperature=0.1, 
                                                    max_decode_steps=1024, 
                                                    top_p=0.8, 
                                                    top_k=40, 
                                                    content=content,
                                                    location = "us-central1",
                                                    )
        # print(textwrap.fill(res, width=100))
        self.palm2_response_time = datetime.now() - _start
        return res
    
    #---CLAUDE-RESPONSE-----------------------------------------------
    def responding_claude(self):
        _start = datetime.now()
        client = Anthropic(api_key=self.ANTHROPIC_API_KEY)
        HUMAN_PROMPT = f"\n\nHuman: {self.prompt}"
        AI_PROMPT = "\n\nAssistant:"
        completion = client.completions.create(
            model="claude-1",
            max_tokens_to_sample=2000,
            temperature=0.1,
            prompt=f"{HUMAN_PROMPT} {AI_PROMPT}",
        )
        
        self.results = completion.completion
        self.claude_response_time = (datetime.now() - _start)
        
        return self.results, self.claude_response_time
        
