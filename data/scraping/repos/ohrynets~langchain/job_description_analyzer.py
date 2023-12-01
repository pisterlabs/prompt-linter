import pandas
import json
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.prompts.chat import HumanMessagePromptTemplate
from pydantic import BaseModel, Field, validator
from langchain.output_parsers import PydanticOutputParser
from typing import Optional
from langchain.output_parsers import RetryWithErrorOutputParser
from langchain.schema import OutputParserException
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate

class JobDescritionStore:
    j = []
    def init(self):
        self.j = []

    def load_csv(self, data_name='data.csv'):
    #loads the csc file and adds the discriptions into a list
        df = pandas.read_csv(data_name)
        for index, row in df.iterrows():
            desc = row['Job Description']
            new_desc = Job_descritor(desc)
            self.j.append(new_desc)
        return(self.j) 

    def save_json(self, file_name='data.json'):
    #saves json file
        desc_list = []
        for item in desc_list:
            desc_list.append(item.__dict__)
        with open(file_name, "w") as fp:
            json.dump(desc_list, fp, indent=4)   
        return 


class Job_descritor:
#defines the lists of the soft, tech, and technologies skills
    def __init__(self, desc):
        self.desc = desc
        self.soft_skills = []
        self.tech_skills = []
        self.technologies = []
   
    def get_desc(self):
        return self.desc
   
    def get_soft_skills(self):
        return self.soft_skills
   
    def get_tech_skills(self):
        return self.tech_skills

    def get_technologies(self):
        return self.technologies

    def set_soft_skills(self, soft_skills):
        self.soft_skills = soft_skills

    def set_tech_skills(self, tech_skills):
        self.tech_skills = tech_skills

    def set_technologies(self, technologies):
        self.technologies = technologies
    


class Job_descriptor_output(BaseModel):
#formats the lists
    tech_skills: Optional[list] = None #= Field(description="list of techincal skills")
    soft_skills: Optional[list] = None #= Field(description="list of soft_skills")
    technologies: Optional[list] = None #= Field(description="list of technologies")


class JobProcessor:
    def __init__(self, open_api_key, temperature, model, template):
    #opens the llm and opens the human message prompt
        self.open_api_key = open_api_key
        self.temperature = temperature
        self.model = model
        self.llm = OpenAI(openai_api_key = openai_api_key, temperature = temperature)
        self.template = template
        self.hm = None #HumanMessagePromptTemplate.from_template(template)

    def get_llm(self):
        return self.llm
    
    def get_hm(self):
        return self.hm
    
    def set_template(self):
        return self.template
    
    #def job_finder(self, job_description):
        #with get_openai_callback() as cb:
            #hms = self.get_hm()
            #llms = self.get_llm()
            #parser = PydanticOutputParser(pydantic_object=Job_descriptor_output)
            #result = llms.predict_messages(
                #[hms.format(job_description=job_description.desc)])
            #out = parser.parse(result.content)
            #return out, cb
        
    def job_finder_1(self, job_description):
    #opens the parser and goes throught the list of Job Descriptions then it defines the prompt
    #and sends it to the llm to get an answer from chatgpt, the retry parser looks for mistakes and
    #reruns the code if it find it. The try function tells it to keep trying untill it fails
        with get_openai_callback() as cb:
            llms = self.get_llm()
            parser = PydanticOutputParser(pydantic_object=Job_descriptor_output)
            prompt = PromptTemplate(
                template=self.template,
                input_variables=["job_description"],
                partial_variables={"format_instructions": parser.get_format_instructions()},
                )
            _input = prompt.format_prompt(job_description=job_description.desc)
            
            #retry_parser = RetryWithErrorOutputParser.from_llm(
                #parser=parser, llm=llms
            #)
            output_parser = CommaSeparatedListOutputParser()
            format_instructions = output_parser.get_format_instructions()
            prompt = PromptTemplate(
                template="List five {subject}.\n{format_instructions}",
                input_variables=["subject"],
                partial_variables={"format_instructions": format_instructions}
                )
            _input = prompt.format(subject="soft skills")
            output = model(_input)
            output = llms(_input.to_string())
            #job_parcer_output = retry_parser.parse(output)
            try:
                job_parcer_output = retry_parser.parse_with_prompt(output, _input)                
            except OutputParserException as e:
                print(f"Unable to parse Job Description:{e}")
                return None, cb
            return job_parcer_output, cb
        



if __name__ == "__main__":
    with open('env.json') as json_file:
        data = json.load(json_file)
        openai_api_key = data['openai_api_key']
        temperature = data['openai_temperature']
        model = data['openai_model']
    template = """/
    Please extract the technical skills from the following job description and store it tech_skills.
    Please provide the output in JSON format and make sure the file is well formatted.
    \n{format_instructions}\n

    {job_description}
    """
    
    job_desc_store = JobDescritionStore()
    job_desc_store.load_csv()
    procs = JobProcessor(openai_api_key, temperature, model, template)
    list_jobs = []
    for skill in job_desc_store.j[:8]:
        skills = procs.job_finder_1(skill)
        if skills is not None:
            list_jobs.append(skills[0].__dict__)
    print(list_jobs)
    with open("analyzed_skills.json", "w") as fp:
        json.dump(list_jobs, fp, indent=4)
    job_desc_store.save_json()
