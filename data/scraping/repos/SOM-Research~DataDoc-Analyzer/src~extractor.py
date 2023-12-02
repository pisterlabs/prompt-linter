import openai
import gradio as gr
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.docstore.document import Document
import pandas as pd
import os
import scipdf ## You need a Gorbid service available
import tabula ## You need to have the Java Tabula installed in the environment
from gradio import DataFrame
import asyncio
from transformers import pipeline
from dotenv import load_dotenv


class Extractor:
    def __init__(self):
        print("Initializing extractor")
        # Init classifier for the post-processing stage
        self.classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-1")

    async def extraction(self, file_name, file_path, apikey, dimension):
        # Build the chains
        chain_incontext, chain_table = self.build_chains(apikey) 
        # Prepare the data
        docsearch = await self.prepare_data(file_name, file_path, chain_table, apikey)
        # Extract dimensions
        if (dimension == "annotation"): 
            results, completeness_report = await self.get_annotation_dimension(docsearch,chain_incontext, retrieved_docs=10)
        elif (dimension == "gathering"):
            results, completeness_report = await self.get_gathering_dimension(docsearch,chain_incontext, retrieved_docs=10)
        elif (dimension == "uses"):
            results, completeness_report = await self.get_uses_dimension(docsearch,chain_incontext, retrieved_docs=10)
        elif (dimension == "contrib"):
            results, completeness_report = await self.get_contributors_dimension(docsearch,chain_incontext, retrieved_docs=10)
        elif (dimension == "comp"):
            results, completeness_report = await self.get_composition_dimension(docsearch,chain_incontext, retrieved_docs=10)
        elif (dimension == "social"):
            results, completeness_report = await self.get_social_concerns_dimension(docsearch,chain_incontext, retrieved_docs=10)
        elif (dimension == "dist"):
            results, completeness_report = await self.get_distribution_dimension(docsearch,chain_incontext, retrieved_docs=10)
        # Get completeness report
        #completeness_report = extractor.postprocessing(results)
        return results, completeness_report
    
    async def complete_extraction(self, file_name, file_path, apikey):
        # Build the chains
        chain_incontext, chain_table = self.build_chains(apikey=apikey) 
        # Prepare the data
        docsearch = await self.prepare_data(file_name, file_path, chain_table, apikey=os.getenv("OPEN_AI_API_KEY"))
        #Retrieve dimensions    
        results = await asyncio.gather(self.get_annotation_dimension(docsearch,chain_incontext, retrieved_docs=10),
                                    self.get_gathering_dimension(docsearch,chain_incontext, retrieved_docs=10),
                                    self.get_uses_dimension(docsearch,chain_incontext, retrieved_docs=10),
                                    self.get_contributors_dimension(docsearch,chain_incontext, retrieved_docs=10),
                                    self.get_composition_dimension(docsearch,chain_incontext, retrieved_docs=10),
                                    self.get_social_concerns_dimension(docsearch,chain_incontext, retrieved_docs=10),
                                    self.get_distribution_dimension(docsearch,chain_incontext, retrieved_docs=10))
        return results
    
    # Extract text from PDF file using SCIPDF and Gorbid service (you need gorbid to use it)
    def extract_text_from_pdf(self, file_path):

        article_dict = scipdf.parse_pdf_to_dict(file_path, soup=True,return_coordinates=False, grobid_url="https://kermitt2-grobid.hf.space") # return dictionary
        print("PDF parsed")
        finaltext = article_dict['title'] + " \n\n " + article_dict['authors'] + " \n\n Abstract: " + article_dict['abstract'] + " \n\n "
        for section in article_dict['sections']:
            sec = section['heading'] + ": "
            if(isinstance(section['text'], str)):
                finaltext = finaltext + sec + section['text'] + " \n\n " 
            else:
                for text in section['text']:
                    sec = sec + text+ " \n\n " 
                finaltext = finaltext + sec
        return finaltext

    # Extract and transform the tables of the papers
    async def get_tables(self, docsearch,chain_table,input_file):   
        print("Getting tables")
        table_texts = []
        dfs = tabula.read_pdf(input_file, pages='all')
        for idx, table in enumerate(dfs):
            query = "Table "+str(idx+1)+":"
            docs = docsearch.similarity_search(query, k=4)
            #result = chain_table({"context":docs,"table":table})
            table_texts.append(self.async_table_generate(docs, table,  chain_table))
            #print(query + " "+ result['text'])
            #table_texts.append(query + " "+ result['text'])
        table_texts = await asyncio.gather(*table_texts)
        for table in table_texts:
            docsearch.add_texts(table[1])
        return docsearch

    def extract_text_clean(self, file_name, file_path):
        file_extension = os.path.splitext(file_name)[1]
        if file_extension == ".pdf":
            all_text = self.extract_text_from_pdf(file_path)
            return all_text
        elif file_extension == ".txt":
            with open(file_path) as f:
                all_text = f.read()
                return all_text

    async def prepare_data(self, file_name, file_path, chain_table, apikey):
        # Process text and get the embeddings
        vectorspath = "./vectors/"+file_name

            #apikey = openai.api_key
        embeddings = OpenAIEmbeddings(openai_api_key=apikey)
        if os.path.isfile(vectorspath+"/index.faiss"):

            # file exists
            docsearch = FAISS.load_local(vectorspath,embeddings=embeddings)

            print("We get the embeddings from local store")
        else:
            #progress(0.40, desc="Detected new document. Splitting and generating the embeddings")
            print("We generate the embeddings using thir-party service")
            # Get extracted running text
            text = self.extract_text_clean(file_name, file_path)

            # Configure the text splitter and embeddings
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=10, separators=[".", ",", " \n\n "])

            # Split, and clean
            texts = text_splitter.split_text(text)
            for idx, text in enumerate(texts):
                texts[idx] = text.replace('\n',' ')
            print("Creating embeddings")
            # Create an index search    
            docsearch = FAISS.from_texts(texts, embeddings, metadatas=[{"source": i} for i in range(len(texts))])
            
            # Extract and prepare tables
        # progress(0.60, desc="Embeddings generated, parsing and transforming tables")
            if (os.path.splitext(file_name)[1] == '.pdf'):
                docsearch = await self.get_tables(docsearch,chain_table,file_path)
            
            # Save the index locally
            FAISS.save_local(docsearch, "./vectors/"+file_name)
    
        return docsearch

    def build_chains(self, apikey):
        LLMClient = OpenAI(model_name='text-davinci-003',openai_api_key=apikey,temperature=0)
        ## In-context prompt
        prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Question: {question}
        ###
        Context: 
        {context}
        ###
        Helpful answer:
        """
        in_context_prompt = PromptTemplate(
            input_variables=["context","question"],
            template=prompt_template,
        )
        chain_incontext = load_qa_chain(LLMClient, chain_type="stuff", prompt=in_context_prompt)

        # Table extraction prompts
        ## Table prompt to transform parsed tables in natural text
        prompt_template = """Given the following table in HTML, and the given context related the table: Translate the content of the table into natural language.
        ###
        Context: 
        {context}
        ###
        Table: {table}
        ###
        Table translation:
        """
        table_prompt = PromptTemplate(
            input_variables=["context","table"],
            template=prompt_template,
        )
        chain_table = LLMChain(llm=LLMClient, prompt=table_prompt)

        return chain_incontext, chain_table

    async def async_table_generate(self, docs,table,chain):

        resp = await chain.arun({"context": docs, "table": table})
        #resp = "Description of the team, the type, and the demographics information, Description of the team, the type, and the demographics information"
        return resp

    async def async_generate(self, dimension, docs,question,chain):
        resp = await chain.arun({"input_documents": docs, "question": question})
        #resp = "Description of the team, the type, and the demographics information, Description of the team, the type, and the demographics information"
        return [dimension, resp]

    async def get_gathering_dimension(self, docsearch, incontext_prompt, retrieved_docs):
        dimensions = [
                    {"Gathering description":"""Provide a summary of how the data of the dataset has been collected? Please avoid mention the annotation process or data preparation processes"""},
                    {"Gathering type":"""Which of the following types corresponds to the gathering process mentioned in the context?

    Types: Web API, Web Scrapping, Sensors, Manual Human Curator, Software collection, Surveys, Observations, Interviews, Focus groups, Document analysis, Secondary data analysis, Physical data collection, Self-reporting, Experiments, Direct measurement, Interviews, Document analysis, Secondary data analysis, Physical data collection, Self-reporting, Experiments, Direct measurement, Customer feedback data, Audio or video recordings, Image data, Biometric data, Medical or health data, Financial data, Geographic or spatial data, Time series data, User-generated content data.

    Answer with "Others", if you are unsure. Please answer with only the type"""},
                    {"Gathering team": """Who was the team who collect the data?"""},
                    {"Team Type": """The data was collected by an internal team, an external team, or crowdsourcing team?""" },
                    {"Team Demographics": "Are the any demographic information of team gathering the data?"},
                    {"Timeframe":""" Which are the timeframe when the data was collected?
                        If present, answer only with the collection timeframe of the data. If your are not sure, or there is no mention, just answers 'not provided'"""},
                    {"Sources": """Which is the source of the data during the collection process? Answer solely with the name of the source""" },
                    {"Infrastructure": """Which tools or infrastructure has been used during the collection process?"""},
                    {"Localization": """Which are the places where data has been collected?
                    If present, answer only with the collection timeframe of the data. If your are not sure, or there is no mention, just answers 'not provided'"""}
        
        ]

        results = []
        for dimension in dimensions:
            for title, question in dimension.items():
                docs = docsearch.similarity_search(question, k=retrieved_docs)
                results.append(self.async_generate(title, docs,question,incontext_prompt))

        answers = await asyncio.gather(*results)

        report = []
        for result in answers:
            if(result[0] == "Gathering description"):
                classifications = self.classifier(result[1], ["Is a description of a process","do not know"])
                if(classifications['labels'][0] == 'Do not know'):
                    print("Dimension: "+result[0]+" is missing. Inserting a warning")
                    report.append(result[0]+" is missing. Please provide an explanation of the data collection process")
            if(result[0] == "Gathering type"):
                classifications = self.classifier(result[1], ["Is others","Is not others"])
                if(classifications['labels'][0] == 'Is others'):
                    print("Dimension: "+result[0]+" is missing. Inserting a warning")
                    report.append(result[0]+" is missing. The type cannot be inferred. Provide a better explanation of the gathering process")
            if(result[0] == "Gathering team"):
                classifications = self.classifier(result[1], ["Is a explanation of a team","Do not know"])
                if(classifications['labels'][0] == 'Do not know'):
                    print("Dimension: "+result[0]+" is missing. Inserting a warning")
                    report.append(result[0]+" is missing. This information is relevant to evaluate the quality of the data")
            if(result[0] == "Team Type"):
                classifications = self.classifier(result[1], ["Is intenal, external or crowdsourcing","Do not know"])
                if(classifications['labels'][0] == 'Do not know'):
                    print("Dimension: "+result[0]+" is missing. Inserting a warning")
                    report.append(result[0]+" is missing. This information is relevant to evaluate the quality of the data")
            if(result[0] == "Team Demographics"):
                classifications = self.classifier(result[1], ["Have demographics information","Do not have demographics information"])
                if(classifications['labels'][0] == 'Do not have demographics information'):
                    print("Dimension: "+result[0]+" is missing. Inserting a warning")
                    report.append(result[0]+" is missing. This information is relevant to evaluate the quality of the labels")
            if(result[0] == "Localization"):  
                classifications = self.classifier(result[1], ["Where data has been collected","unknown"])
                if(classifications['labels'][0] == 'Is not a localization'):
                    print("Dimension: "+result[0]+" is missing. Inserting a warning")
                    report.append(result[0]+" is missing. Please indicate where the data has been collected")
            if(result[0] == "Timeframe"):  
                classifications = self.classifier(result[1], ["It is a date","It is not a date"])
                if(classifications['labels'][0] == 'It is not a date'):
                    print("Dimension: "+result[0]+" is missing. Inserting a warning")
                    report.append(result[0]+" is missing. Please indicate when the data has been collected")
            if(result[0] == "Infrastructure"):  
                classifications = self.classifier(result[1], ["Is a tool or an infrastructure","Is not a tool or an infrastructure"])
                if(classifications['labels'][0] == 'Is not a tool or an infrastructure'):
                    print("Dimension: "+result[0]+" is missing. Inserting a warning")
                    report.append(result[0]+" is missing. Please indicate the infrastructure used to collect the data")
            if(result[0] == "Sources"):  
                classifications = self.classifier(result[1], ["Is source of data","Do not know"])
                if(classifications['labels'][0] == 'Do not know'):
                    print("Dimension: "+result[0]+" is missing. Inserting a warning")
                    report.append(result[0]+" is missing. Please indicate the source used to collect the data")
        if len(report) == 0:
            report.append("No warnings")
            completeness = 100
        else:
            completeness = round((1 - len(report)/len(answers))*100)
        completeness_report = {"completeness":completeness,"report":report}
        return answers, completeness_report

    async def get_annotation_dimension(self, docsearch, incontext_prompt, retrieved_docs):
        dimensions = [
                    {"Annotation description":"""How the data of the has been annotated or labelled? Provide a short summary of the annotation process"""},
                    {"Annotation type":""" Which  of the following category corresponds to the annotation process mentioned in the context? 
                        Categories: Bounding boxes, Lines and splines, Semantinc Segmentation, 3D cuboids, Polygonal segmentation, Landmark and key-point, Image and video annotations, Entity annotation, Content and textual categorization
                        If you are not sure, answer with 'others'. Please answer only with the categories provided in the context. """},
                    {"Labels":""" Which are the specific labels of the dataset? Can you enumerate it an provide a description of each one?"""},
                    {"Team Description": """Who has annotated the data?"""},
                    {"Team type": """The data was annotated by an internal team, an external team, or crowdsourcing team?""" },
                    {"Team Demographics": """Is there any demographic information about the team who annotate the data?"""},
                    {"Infrastructure": """Which tool has been used to annotate or label the dataset?"""},
                    {"Validation": """How the quality of the labels have been validated?""" }
        ]

        results = []
        for dimension in dimensions:
            for title, question in dimension.items():
              docs = docsearch.similarity_search(question, k=retrieved_docs)
              results.append(self.async_generate(title, docs,question,incontext_prompt))

        answers = await asyncio.gather(*results)

        ## Post-processing
        report = []
        for result in answers: 
             if(result[0] == "Annotation Description"):
                classifications = self.classifier(result[1], ["Is a description of a process","Is unknown"])
                if(classifications['labels'][0] == 'Is unkown'):
                    print("Dimension: "+result[0]+" is missing in the documentation")
                    report.append(result[0]+" is missing. Please, provide a better explanation of the annotation process")
             if(result[0] == "Annotation Type"):
                classifications = self.classifier(result[1], ["Is others","Is not others"])
                if(classifications['labels'][0] == 'Is others'):
                    print("Dimension: "+result[0]+" is missing in the documentation")
                    report.append(result[0]+" is missing. The type of the annotation process cannot be infered form the documentation. Please, provide a better explanation of the process")
             if(result[0] == "Labels"):
                classifications = self.classifier(result[1], ["Labels explanation","do not know"])
                if(classifications['labels'][0] == 'do not know'):
                    print("Dimension: "+result[0]+" is missing in the documentation")
                    report.append(result[0]+" is missing. Please provide a better explanation of the labels generated with the annotation process")
             if(result[0] == "Team Description"):
                classifications = self.classifier(result[1], ["Is a description of a team","Is not a description of a team"])
                if(classifications['labels'][0] == 'Is not a description of a team'):
                    print("Dimension: "+result[0]+" is missing in the documentation")
                    report.append(result[0]+" is missing. This information is relevant to evaluate the quality of the labels")
             if(result[0] == "Team Type"):
                classifications = self.classifier(result[1], ["Is intenal, external or crowdsourcing","Do not know"])
                if(classifications['labels'][0] == 'Do not know'):
                    print("Dimension: "+result[0]+" is missing. Inserting a warning")
                    report.append(result[0]+" is missing. This information is relevant to evaluate the quality of the data")
             if(result[0] == "Team Demographics"):
                classifications = self.classifier(result[1], ["Have demographics information","Do not have demographics information"])
                if(classifications['labels'][0] == 'Do not have demographics information'):
                    print("Dimension: "+result[0]+" is missing. Inserting a warning")
                    report.append(result[0]+" is missing. This information is relevant to evaluate the quality of the labels")
             if(result[0] == "Infrastructure"):  
                classifications = self.classifier(result[1], ["Is a tool or an infrastructure","Is not a tool or an infrastructure"])
                if(classifications['labels'][0] == 'Is not a tool or an infrastructure'):
                    print("Dimension: "+result[0]+" is missing. Inserting a warning")
                    report.append(result[0]+" is missing. Please indicate the infrastructure used to annotate the data")
             if(result[0] == "Validation"):  
                classifications = self.classifier(result[1], ["Is there a method","It is not a method"])
                if(classifications['labels'][0] == 'Is is not a method'):
                    print("Dimension: "+result[0]+" is missing. Inserting a warning")
                    report.append(result[0]+" is missing. Please indicate how the annotation have been validated")        
        if len(report) == 0:
            report.append("No warnings")
            completeness = 100
        else:
            completeness = round((1 - len(report)/len(answers))*100)
        completeness_report = {"completeness":completeness,"report":report}

        return answers, completeness_report
    
    async def get_social_concerns_dimension(self, docsearch, incontext_prompt, retrieved_docs):
        dimensions = [
                    {"Representativeness":"""Are there any social group that could be misrepresented in the dataset?"""},
                    {"Biases":"""Is there any potential bias or imbalance in the data?"""},
                    {"Sensitivity":""" Are there sensitive data, or data that can be offensive for people in the dataset?"""},
                    {"Privacy":""" Is there any privacy issues on the data?"""},
                    
        ]

        results = []
        for dimension in dimensions:
            for title, question in dimension.items():
              docs = docsearch.similarity_search(question, k=retrieved_docs)
              results.append(self.async_generate(title, docs,question,incontext_prompt))

        answers = await asyncio.gather(*results)

        ## Post-processing
        report = []
        for result in answers: 
             if(result[0] == "Representativeness"):
                classifications = self.classifier(result[1], ["Representativeness","Not mentioned or do not know"])
                if(classifications['labels'][0] == 'Not mentioned or do not know'):
                    print("Dimension: "+result[0]+" is missing in the documentation")
                    report.append(result[0]+" is missing. Is there any representativeness issue in your data?")
             if(result[0] == "Biases"):
                classifications = self.classifier(result[1], ["Is a bias explanation","Not mentioned or do not know"])
                if(classifications['labels'][0] == 'Not mentioned or do not know'):
                    print("Dimension: "+result[0]+" is missing in the documentation")
                    report.append(result[0]+" is missing. Are you sure there is no potential bias in your data?")
             if(result[0] == "Sensitivity"):
                classifications = self.classifier(result[1], ["Explanation of sensibilty data issue","Not mentioned or do not know"])
                if(classifications['labels'][0] == 'Not mentioned or do not know'):
                    print("Dimension: "+result[0]+" is missing in the documentation")
                    report.append(result[0]+" is missing. Are you sure there is no sensitivity data in your dataset?")
             if(result[0] == "Privacy"):
                classifications = self.classifier(result[1], ["Is privacy issue","Not mentioned or do not know"])
                if(classifications['labels'][0] == "Not mentioned or do not know"):
                    print("Dimension: "+result[0]+" is missing in the documentation")
                    report.append(result[0]+" is missing. Are you sure there is no privacy issues in your data?")
        if len(report) == 0:
           report.append("No warnings")
           completeness = 100
        else:
           completeness = round((1 - len(report)/len(answers))*100)
        completeness_report = {"completeness":completeness,"report":report}

        return answers, completeness_report

    async def get_uses_dimension(self, docsearch, incontext_prompt, retrieved_docs):
        dimensions = [
                    {"Purposes":"""Which are the purpose or purposes of the dataset?"""},
                    {"Gaps":"""Which are the gaps the  dataset intend to fill?"""},
                    {"Task":"""Which machine learning tasks the dataset inteded for?:"""},
                    {"Recommended":"""For which applications the dataset is recommended?"""},
                    {"Non-Recommneded":"""Is there any non-recommneded application for the dataset? If you are not sure, or there is any non-recommended use of the dataset metioned in the context, just answer with "no"."""},
        ]
        results = []
        for dimension in dimensions:
            for title, question in dimension.items():
              docs = docsearch.similarity_search(question, k=retrieved_docs)
              if (title == "Task"):
                question = """Which of the following ML tasks for the dataset best matches the context?  
                
                        Tasks: text-classification, question-answering, text-generation, token-classification, translation,
                        fill-mask, text-retrieval, conditional-text-generation, sequence-modeling, summarization, other,
                        structure-prediction, information-retrieval, text2text-generation, zero-shot-retrieval,
                        zero-shot-information-retrieval, automatic-speech-recognition, image-classification, speech-processing,
                        text-scoring, audio-classification, conversational, question-generation, image-to-text, data-to-text,
                        classification, object-detection, multiple-choice, text-mining, image-segmentation, dialog-response-generation,
                        named-entity-recognition, sentiment-analysis, machine-translation, tabular-to-text, table-to-text, simplification,
                        sentence-similarity, zero-shot-classification, visual-question-answering, text_classification, time-series-forecasting,
                        computer-vision, feature-extraction, symbolic-regression, topic modeling, one liner summary, email subject, meeting title,
                        text-to-structured, reasoning, paraphrasing, paraphrase, code-generation, tts, image-retrieval, image-captioning,
                        language-modelling, video-captionning, neural-machine-translation, transkation, text-generation-other-common-sense-inference,
                        text-generation-other-discourse-analysis, text-to-tabular, text-generation-other-code-modeling, other-text-search

                        If you are not sure answer with just with "others".
                        Please, answer only with one or some of the provided tasks """
                
              results.append(self.async_generate(title, docs,question,incontext_prompt))

        answers = await asyncio.gather(*results)
        ## Post-processing
        report = []
        for result in answers: 
             if(result[0] == "Purposes"):
                classifications = self.classifier(result[1], ["Is there purposes","Not mentioned or do not know"])
                if(classifications['labels'][0] == 'Not mentioned or do not know'):
                    print("Dimension: "+result[0]+" is missing in the documentation")
                    report.append(result[0]+" is missing. Please provide a better explanation of the purposes of the dataset")
             if(result[0] == "Gaps"):
                classifications = self.classifier(result[1], ["Gaps the dataset intends to fill","Not mentioned or do not know"])
                if(classifications['labels'][0] == 'Not mentioned or do not know'):
                    print("Dimension: "+result[0]+" is missing in the documentation")
                    report.append(result[0]+" is missing. Which gaps this dataset intends to fill?")
             if(result[0] == "Task"):
                classifications = self.classifier(result[1], ["Is a task","Others"])
                if(classifications['labels'][0] == 'Others'):
                    print("Dimension: "+result[0]+" is missing in the documentation")
                    report.append(result[0]+" is missing. The task of the dataset cannot be inferred, please provide a better explanation of its purposes?")
             if(result[0] == "Recommended"):
                classifications = self.classifier(result[1], ["Is a recommendation","Not mentioned or do not know"])
                if(classifications['labels'][0] == "Not mentioned or do not know"):
                    print("Dimension: "+result[0]+" is missing in the documentation")
                    report.append(result[0]+" is missing. Which are the uses recommendation of your dataset?")
             if(result[0] == "Non-Recommneded"):
                classifications = self.classifier(result[1], ["Is a non-recommneded use","No non-recommended use"])
                if(classifications['labels'][0] == "No non-recommended use"):
                    print("Dimension: "+result[0]+" is missing in the documentation")
                    report.append(result[0]+" is missing. Is there any non-recommended use of the data?")
        if len(report) == 0:
           report.append("No warnings")
           completeness = 100
        else:
           completeness = round((1 - len(report)/len(answers))*100)
        completeness_report = {"completeness":completeness,"report":report}

        return answers, completeness_report

    async def get_contributors_dimension(self, docsearch, incontext_prompt, retrieved_docs):
        dimensions = [
                    {"Authors":"""Who are the authors of the dataset """},
                    {"Funders":"""Is there any organization which supported or funded the creation of the dataset?"""},
                    {"Maintainers":"""Who are the maintainers of the dataset?"""},
                    {"Erratums":"""Is there any data retention limit in the dataset? If you are not sure, or there is no retention limit just answer with "no"."""},
                    {"Data Retention Policies":"""Is there any data retention policies policiy of the dataset? If you are not sure, or there is no retention policy just answer with "no"."""},
        ]

        results = []
        for dimension in dimensions:
            for title, question in dimension.items():
              docs = docsearch.similarity_search(question, k=retrieved_docs)
              results.append(self.async_generate(title, docs,question,incontext_prompt))

        answers = await asyncio.gather(*results)
          ## Post-processing
        report = []
        for result in answers: 
             if(result[0] == "Authors"):
                classifications = self.classifier(result[1], ["Authors","Not mentioned or do not know"])
                if(classifications['labels'][0] == 'Not mentioned or do not know'):
                    print("Dimension: "+result[0]+" is missing in the documentation")
                    report.append(result[0]+" is missing. Authors cannot be identified")
             if(result[0] == "Funders"):
                classifications = self.classifier(result[1], ["Funders of the dataset","Not mentioned or do not know"])
                if(classifications['labels'][0] == 'Not mentioned or do not know'):
                    print("Dimension: "+result[0]+" is missing in the documentation")
                    report.append(result[0]+" is missing. Who funded the dataset?")
             if(result[0] == "Mantainers"):
                classifications = self.classifier(result[1], ["Maintainers","Not mentioned or do not know"])
                if(classifications['labels'][0] == 'Not mentioned or do not know'):
                    print("Dimension: "+result[0]+" is missing in the documentation")
                    report.append(result[0]+" is missing. Who were the maintainers of the dataset?")
             if(result[0] == "Erratums"):
                classifications = self.classifier(result[1], ["Is an Erratum","No erratum"])
                if(classifications['labels'][0] == "No erratum"):
                    print("Dimension: "+result[0]+" is missing in the documentation")
                    report.append(result[0]+" is missing. Is there an erratum?")
             if(result[0] == "Data Retention Policies"):
                classifications = self.classifier(result[1], ["Data Retention","No data retention policy"])
                if(classifications['labels'][0] == "No data retention policy"):
                    print("Dimension: "+result[0]+" is missing in the documentation")
                    report.append(result[0]+" is missing. Is there any data retention policy  of the data?")
        if len(report) == 0:
           report.append("No warnings")
           completeness = 100
        else:
           completeness = round((1 - len(report)/len(answers))*100)
        completeness_report = {"completeness":completeness,"report":report}

        return answers, completeness_report

    async def get_composition_dimension(self, docsearch, incontext_prompt, retrieved_docs):
        dimensions = [
                    {"File composition":"""Can you provide a description of each files the dataset is composed of?"""},
                    {"Attributes":"""Can you enumerate the different attributes present in the dataset? """},
                    {"Training splits":"""The paper mentions any recommended data split of the dataset?"""},
                    {"Relevant statistics":"""Are there relevant statistics or distributions of the dataset? """},
        ]

        results = []
        for dimension in dimensions:
            for title, question in dimension.items():
              docs = docsearch.similarity_search(question, k=retrieved_docs)
              results.append(self.async_generate(title, docs,question,incontext_prompt))

        answers = await asyncio.gather(*results)

        ## Post-processing
        report = []
        for result in answers: 
             if(result[0] == "File composition"):
                classifications = self.classifier(result[1], ["A file composition","Not mentioned or do not know"])
                if(classifications['labels'][0] == 'Not mentioned or do not know'):
                    print("Dimension: "+result[0]+" is missing in the documentation")
                    report.append(result[0]+" is missing. Provide a better explanation of the file composition of the dataset")
             if(result[0] == "Attributes"):
                classifications = self.classifier(result[1], ["Attributes explanation","Not mentioned or do not know"])
                if(classifications['labels'][0] == 'Not mentioned or do not know'):
                    print("Dimension: "+result[0]+" is missing in the documentation")
                    report.append(result[0]+" is missing. Provide a better explanation of the attribute explanation of the dataset")
             if(result[0] == "Training splits"):
                classifications = self.classifier(result[1], ["A data split","Not mentioned or do not know"])
                if(classifications['labels'][0] == 'Not mentioned or do not know'):
                    print("Dimension: "+result[0]+" is missing in the documentation")
                    report.append(result[0]+" is missing. Is there any recommended data split?")
             if(result[0] == "Relevant statistics"):
                classifications = self.classifier(result[1], ["A statistic","Not mentioned or do not know"])
                if(classifications['labels'][0] == "Not mentioned or do not know"):
                    print("Dimension: "+result[0]+" is missing in the documentation")
                    report.append(result[0]+" is missing. Is there any relevant statistic?")
        if len(report) == 0:
           report.append("No warnings")
           completeness = 100
        else:
           completeness = round((1 - len(report)/len(answers))*100)
        completeness_report = {"completeness":completeness,"report":report}

        return answers, completeness_report

    async def get_distribution_dimension(self, docsearch, incontext_prompt, retrieved_docs):
        dimensions = [
                    {"Data repository":"""Is there a link to the a repository containing the data? If you are not sure, or there is no link to the repository just answer with "no"."""},
                    {"Licence":"""Which is the license of the dataset. If you are not sure, or there is mention to a license of the dataset in the context, just answer with "no". """},
                    {"Deprecation policies":"""Is there any deprecation plan or policy of the dataset?
                    """},
                    
        ]

        results = []
        for dimension in dimensions:
            for title, question in dimension.items():
              docs = docsearch.similarity_search(question, k=retrieved_docs)
              results.append(self.async_generate(title, docs,question,incontext_prompt))

        answers = await asyncio.gather(*results)
         ## Post-processing
        report = []
        for result in answers: 
             if(result[0] == "Data repository"):
                classifications = self.classifier(result[1], ["A link to a repository","Not mentioned or do not know"])
                if(classifications['labels'][0] == 'Not mentioned or do not know'):
                    print("Dimension: "+result[0]+" is missing in the documentation")
                    report.append(result[0]+" is missing. Where the data can be accessed?")
             if(result[0] == "Licence"):
                classifications = self.classifier(result[1], ["A License","Not mentioned or do not know"])
                if(classifications['labels'][0] == 'Not mentioned or do not know'):
                    print("Dimension: "+result[0]+" is missing in the documentation")
                    report.append(result[0]+" is missing. Which is the license of the dataset")
             if(result[0] == "Deprecation policies"):
                classifications = self.classifier(result[1], ["A deprecation policy","No a deprecation policy"])
                if(classifications['labels'][0] == 'No a deprecation policy'):
                    print("Dimension: "+result[0]+" is missing in the documentation")
                    report.append(result[0]+" is missing. Is there any deprecation policy of the dataset?")
        if len(report) == 0:
           report.append("No warnings")
           completeness = 100
        else:
           completeness = round((1 - len(report)/len(answers))*100)
        completeness_report = {"completeness":completeness,"report":report}

        return answers, completeness_report

    