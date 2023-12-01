import csv
import pydantic

from pydantic import Field, BaseModel, validator
from typing import Iterator, Optional
from kor import Object, Text, Number

from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser

import spacy
import pandas as pd
import time
from langchain.llms import OpenAI
import numpy as np
import os
from langchain.text_splitter import SpacyTextSplitter
from kor import create_extraction_chain
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.llms import OpenAI


from negspacy.negation import Negex
from negspacy.termsets import termset


#Format below given clinicalAbbr as key value pairs between abbr and description
abbr = { "CSF": "cerebrospinal fluid", 
                "CSU": "catheter stream urine sample",
                "CT scan": "computerised tomography scan",
                "CVP": "central venous pressure", 
                "CXR": "chest X-ray",
                "DNACPR": "do not attempt cardiopulmonary resuscitation",
                "DNAR": "do not attempt resuscitation",
                "DNR": "do not resuscitate", "Dr": "doctor",
                "DVT": "deep vein thrombosis",
                "Dx": "diagnosis",
                "ECG": "electrocardiogram", 
                "ED": "emergency department"
            }

class patient(pydantic.BaseModel):
    patientName: str = Field(
        description="The name of the patient, First Name and Last Name concatenated",
        examples = "[**First Name (NamePattern)**]"
    )
    diagnosisData: str = Field(
        description="The summary of the diagnosis of the patient or the final diagnosis of the patient",
    )
    time : Optional[str] = Field(
        description="The time of the diagnosis of the patient",
    )
    clinicalAbbr : Optional[str] = Field(
        description="The clinical abbreviations present in the diagnosis of the patient",
        #Write an example for the same including a clinical abbreviation for the model to detect
        examples = abbr
    )
    medsData : Optional[str] = Field(
        description="The medications of the patient prescribed or administered to the patient, might have abbreviations",
        examples = ["Paracetamol 75mcg p.o." , "Aspirin 81mg p.i. q.d."]
    )
    dischargeData : Optional[str] = Field(
        description="The summary of the discharge of the patient, including the medical status of the patient",
        example= "The patient was able to oxygenate on room air at 93% at the time of discharge."
    )
    medicalScans : Optional[list] = Field(
        description="The summary of the medical scans of the patient, including but not limited to CT Scans, MRIs and X-Ray Scans",
        example = [""]
    )
    additionalInfo : Optional[str] = Field(
        description="Any additional information about the patient, including but not limited to the patient's medical history, allergies, etc.",
        example = "The patient has had a history of asthma."
    )

def generatorDecorator(func):
    def wrapper(*args, **kwargs):
        #Return the timing of the execution and the analysis of pipeline
        timer = time.time()
        gen = next(func(*args, **kwargs))
        timerOut = time.time() 
        diffTime = timerOut - timer
        print(f"Time to execute: {diffTime}")
        
        analysis = nlp.analyze_pipes(pretty=True)
        print(analysis)
        return gen
    return wrapper

@generatorDecorator
def generatorRead(Path_csv: str) -> Iterator[list]:
    with open(Path_csv, "r") as f:
        reader = csv.reader(f)
        countRow = 0
        for row in reader:
            if countRow:
                countRow += 1 
                continue
            
            modRow = yield row
            spacyPipe = nlp(modRow[2])
            contextData = spacyPipe.to_dict()
            modRow.append(contextData)

            yield modRow
            countRow += 1

def chunkingFile(totalData: str, chunkSize: int):
    #Chunk using SpaCy TextSplitter
            
            

if __name__=="main":

    #Define the OpenAI API key
    aiKey = "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"
    os.environ["OPENAI_API_KEY"] = aiKey


    #Use langchain openai model
    llm = OpenAI(openai_api_key=aiKey, model="text-davinci-003", temperature=0.1) # type: ignore
    # summaryChain = load_summarize_chain(llm, chain_type="map_reduce", return_intermediate_results=True, map_prompt=prompt, combine_prompt=prompt)


    nlp = spacy.load("en_core_med7_lg" )
    ts = termset("en_clinical")
    textSplit = SpacyTextSplitter()
    


    nlp.add_pipe("entity_linker")
    nlp.add_pipe(
        "negex",
        config={
            "neg_termset":ts.get_patterns()
        }
    )

    csvPath = r"C:\Users\jreno\Documents\Projects\Mycoach Health\NOTEEVENTS\NOTEEVENTS.csv.gz"
    details = generatorRead(csvPath)
    
    #Add generator invocation here
    userInput = int(input("Enter how many records to generator: "))
    for i in range(userInput):
        genRow = next(details)
        
        #Add chunking here
        docPatient = textSplit.split_text(str(genRow[2]))
        docContext = textSplit.split_text(str(genRow[3]))
        
        #Embed the data and then extract json from it
        parser = PydanticOutputParser(pydantic_object=Patient)

        prompt = PromptTemplate(
            template="Answer the user query.\n{format_instructions}\n{query}\n",
            input_variables=["patientData", "query", "contextData],
            partial_variables={"format_instructions": parser.get_format_instructions()},
            )

        _input = prompt.format_prompt(query=medQuery, patientData=docPatient, contextData=docContext)

        output = model(_input.to_string())

        parser.parse(output)
    
    
    
    
