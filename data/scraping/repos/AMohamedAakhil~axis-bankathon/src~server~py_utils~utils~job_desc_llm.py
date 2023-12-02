import os
from langchain.chains import LLMChain, SequentialChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import asyncio
import json
import os
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


class JobDescLLM:
     def __init__(self, job_title, job_description) -> None:
          self.job_title = job_title
          self.job_description = job_description
          self.llm = OpenAI(temperature=0.6, max_tokens=-1, openai_api_key=OPENAI_API_KEY)

          with open("src/server/py_utils/json_components/jd_elements.json") as elements_file:
               self.elements = json.load(elements_file)




     async def build_dict(self):
          prompt1 = """
          Given a Job description for the Job title: {job_title},
          Analyse the description and identify the following component:

         {field}: {field_info}

          Scan the description for the presence of this field, Identify the segment.
          ONLY Return a json formatted string where the property name is {field} and value is the identied segment
          If you cannot identify the segment, The value of the property should be null.
          -----
          Job description :
          {job_description}


          Make sure to RETURN ONLY JSON, not code or any other text. 
          """

          prompt1_template = PromptTemplate(
               input_variables=["job_title" ,"job_description", "field", "field_info"],
               template= prompt1
          )

          chain1 = LLMChain(llm = self.llm, prompt=prompt1_template, 
                            output_key="dictionary"
                        )
          
          dict_chain = SequentialChain(
               chains = [chain1],
               input_variables = ["job_title", "job_description", "field", "field_info"],
               output_variables = ["dictionary"],
               verbose = False
          )
          
          tasks = []
          for field in self.elements.keys():
               field_info = self.elements[field]
               inputs = {
               "job_title": self.job_title,
               "field": field,
               "field_info": field_info,
               "job_description": self.job_description
               }

               tasks.append(self.async_generate(dict_chain, inputs))
          results = await asyncio.gather(*tasks)

          elements_dict = {}
          
          for json_stuff in results:
               temp_ = json.loads(json_stuff)
               name_ = list(temp_.keys())[0]
               
               elements_dict[name_] = temp_[name_]
          return elements_dict

     def create_enhancement_chain(self, inputs):
          enhancements_pt = PromptTemplate(
               input_variables=["job_description", "flag_dict"],
               template = """
          Your boss has written a job description to recruit new employees. 

          ---
          Job description:
          {job_description}
          --
          Flag dictionary:
          {flag_dict}

          YOUR JOB:

          For each field in the flag dictionary, If the flag = 1, 
          Analyse the field and reccomendations enhancements to be added to the field with relevance to the given job.
          
          If the flag = 0, State that NO enhancements has to be added to the field.

          Return the name of the field, and the reccomended enhancements, if any.
          """
          )
          chain = LLMChain(llm = self.llm,
                           prompt=enhancements_pt,
                           output_key="enhancements")
          
          enhance_pt = PromptTemplate(
               input_variables=["enhancements", "job_description", "job_title"],
               template = """
               Your boss has written a job description for the post of {job_title} and 
               your coworker has reccomendatations for enhancements to be incorporated in the job description. 

               For each of the fields present in the job description, It will contain a score and 
               It will be marked as flagged or not flagged. 
               IF the field is flagged, It will contain a reccomendation for enhancement
               ---
               Job description:
               {job_description}
               ---
               Reccomendations for enhancements:
               {enhancements}
               ---
               YOUR JOB:
               1) LOOP through each field present in the reccomendations.
               2) in the field, IF reccomendation is provided, Consider the provided reccomendation and edit the job description accordingly.
               3) If reccomendation is not provided, DO NOT EDIT the field in the job description!!

               Return ONLY the edited FULL job description. 
               """
          )

          chain2 = LLMChain(llm = self.llm,
                           prompt=enhance_pt,
                           output_key="final_job_desc")
          
          sqc = SequentialChain(
               chains = [chain,chain2],
               input_variables=["job_description", "flag_dict", "job_title"],
               output_variables=["enhancements", "final_job_desc"],
               verbose=False
               
          )

          return sqc(inputs)
     

     
     async def async_generate(self, sqc, inputs):
          resp = await sqc.arun(inputs)
          return resp
     
     
     async def generate_concurrently(self):
          pt = PromptTemplate(
               input_variables=["job_title", "field", "field_info", "field_val"],
               template = """
              Your company is recruting employees for the job, {job_title}. Your boss 
              has written the {field} for this job, Your job is to evaluate on How 
              well he has written it. See if the given {field} is actually relevant for 
              the job and whether or not it accurately describes the {field} REQUIRED.
              FOR SOMEBODY TO FUNCTION EFFECTIVELY AS A {job_title}. Based on this, Score the
              {field} out of 10. If the field has been listed as null, give the score as 0
              You are allowed to use decimal values, Return ONLY THE SCORE AND NO OTHER TEXT.

              {field} = {field_info}. 
              ---
              This is what your boss has written:

              {field_val}
              ---         
          """
          )
          chain = LLMChain(llm = self.llm,
                           prompt=pt,
                           output_key="score")
          
          sqc = SequentialChain(
               chains = [chain],
               input_variables = ["job_title", "field", "field_info", "field_val"],
               output_variables = ["score"],
               verbose = False
          )

          elements_dict = await self.build_dict() 
          tasks = []

          for field in elements_dict.keys():
               field_info = self.elements[field]
               field_val = elements_dict[field]
               inputs = {
               "job_title": self.job_title,
               "field": field,
               "field_info": field_info,
               "field_val": field_val
               }

               tasks.append(self.async_generate(sqc, inputs))

          results = await asyncio.gather(*tasks)
          results  = [float(i) for  i in results]

          sum_score = sum(results)
          scoring_dict = {k:v for (k,v) in zip(elements_dict.keys(), results)}
          flag_dict = {}
          threshold = 8.0

          for key in scoring_dict.keys():
               if scoring_dict[key]< threshold:
                    flag_dict[key] = 1
               else:
                    flag_dict[key] = 0
          


          enchancements_out = self.create_enhancement_chain({
               "job_description": self.job_description,
               "job_title": self.job_title,
               "flag_dict": flag_dict
          })

          return sum_score, scoring_dict, enchancements_out['enhancements'], enchancements_out['final_job_desc']

          



     
    








