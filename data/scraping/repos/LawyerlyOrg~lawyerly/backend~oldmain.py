from gpt_search import *
from ingest import *
from fact_sheet import *
from commands import *
from evaluate_cases import *
from db import *
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from pymongo import MongoClient

# Store API keys in OS env
os.environ["OPENAI_API_KEY"] = constants.OPENAI_API_KEY
os.environ["SERPAPI_API_KEY"] = constants.SERPAPI_API_KEY
os.environ["PINECONE_API_KEY"] = constants.PINECONE_API_KEY
os.environ["MONGODB_URI"] = constants.MONGODB_URI

# Initialize OpenAI props
embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
openai.api_key = OPENAI_API_KEY

# Initialize Pinecone index
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment="northamerica-northeast1-gcp"
)
index_name = "test3"

# GLOBAL VARIABLES

case_file_names = ['2012_2RCS_584.pdf', '1990canlii95.pdf', '2017nbca10.pdf', 'Cowan-en.pdf', 'Khill-en.pdf']
collection_name = 'STD Collection'
law_area = 'criminal_law'
index = get_existing_index(index_name, embeddings, collection_name)

sample_fact_sheet = """
• Victim is Allison.  She and her partner are from Canada.  
• Allison is sexually active with her partner the defendant Daniel.  
• Daniel and Allison have been having sex for 14 months.  
• Prior to meeting Allison , Daniel discovered he had syphilis.  
• Daniel did not tell client Allison that he has syphilis , because he thought she would not agree to have sex with him if he did.  
• Daniel and Allison continued having sexual intercourse.  
• Allison has recently discovered that she contracted syphilis from Daniel.  
• Allison would not have continued having sex with Daniel had she known he had 
syphilis. 
"""
sample_summary = """The defendant is accused of failing to disclose his HIV-positive status to nine complainants before having sex with them, which did not result in any of the complainants contracting HIV. The actus reus of the charged crime is failing to disclose one's HIV-positive status to a sexual partner before having sex with them, and the mens rea is intent to deceive."""

### 

def summarize_and_find_relevancies():
    # Step 1: summarize case and get string as output
    summary_string = extract_summary(law_area, index, case_file_names[0])
    # Step 2: find relevancies between summary and factsheet
    relevancies = evaluate_relevancy(sample_fact_sheet, summary_string)
    
    return relevancies


def main():
    return 0
    
if __name__ == '__main__':
    main()
