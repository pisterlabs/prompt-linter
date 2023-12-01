from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from langchain import PromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory
import os
from os.path import dirname, join 
from dotenv import load_dotenv
from VecDB import cossim, Embedder 

dotenv_path = join(dirname(__file__), ".env")
load_dotenv(dotenv_path)
API_TOKEN = os.environ.get("APIKEY")
PROJECT_ID = os.environ.get("PROJECT_ID")
SYSMSG_NOTES = os.environ.get("SYSMSG_NOTES")

class NotesAnalyst():
    def __init__(self):
        self.template = SYSMSG_NOTES
        self.prompt = PromptTemplate(
        input_variables=["note"],
        template=self.template)

        GenParams().get_example_values()

        #model hyperparameters 
        self.generate_params = {
            GenParams.MIN_NEW_TOKENS: 50,
            GenParams.MAX_NEW_TOKENS: 300,
            GenParams.TEMPERATURE: 0.0,
            GenParams.REPETITION_PENALTY: 1,
        }

        #initializing the model 
        self.model = Model(
            model_id=ModelTypes.LLAMA_2_70B_CHAT,
            params=self.generate_params,
            credentials={
                "apikey": API_TOKEN,  
                "url": "https://eu-de.ml.cloud.ibm.com"
            },
            project_id=PROJECT_ID
        )

        self.chain = LLMChain(llm=self.model.to_langchain(), prompt=self.prompt, verbose=False)
        self.embedder = Embedder()

    def analyze(self, note:str):
        analysis = self.chain(note)

        return analysis
    
    def search(self, query:str, notes:list, thresh=0.7):
        query_embedded = self.embedder.get_embedding(query)

        for n in notes:
            cs = cossim(query_embedded, n)
            print(cs)
            if cs >= thresh:
                return n 
            

if __name__ == '__main__':
    analyst  = NotesAnalyst()
    note = 'Today I felt quite happy. I talked to my friends at school and played with my big snow-white dog at home later. Mom cooked salmon for dinner and all of us (my parens, sister and me) ate at the table, while also talking about our day. It was nice.'
    analysis = analyst.analyze(note)
    print(analysis['text'])

    notes = [note]

    query = 'white dog'
    n = analyst.search()