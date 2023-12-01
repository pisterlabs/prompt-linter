from Google import Create_Service
import gspread
import langchain
from langchain.chat_models import ChatOpenAI
import pymysql
from langchain.document_loaders.csv_loader import UnstructuredCSVLoader
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain import PromptTemplate, LLMChain
import os
import csv
from twilio.rest import Client

from dotenv import load_dotenv
pymysql.install_as_MySQLdb()


load_dotenv()

OPENAI_API_TOKEN=os.getenv("OPENAI_API_TOKEN")

os.environ["OPENAI_API_TOKEN"] = OPENAI_API_TOKEN
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="./credentials.json"

# Your Account SID from twilio.com/console
account_sid = os.environ["TWILIO_ACCOUNT_SID"]
    
# Your Auth Token from twilio.com/consoles
auth_token  = os.environ["TWILIO_AUTH_TOKEN"]
    
client = Client(account_sid, auth_token)


chat_llm=ChatOpenAI(model="gpt-4", openai_api_key=OPENAI_API_TOKEN)

#connect to the database 
connection = pymysql.connect(host=os.environ["DB_HOST"],
                                user=os.environ["DB_USERNAME"],
                                password=os.environ["DB_PASSWORD"],
                                database= os.environ["DATABASE"],
                                ssl_ca='./cert.pem' # From AWS's cert.pem
                                )

#using service account credentials json file to instantiate the 
Service_Account=gspread.service_account("credentials.json")

cursor=connection.cursor()


#get all clients
query = """
SELECT pers.phone, pers.fname, pr.programName
FROM Client c 
JOIN Profile pr ON c.profileId = pr.id
JOIN Personal pers ON c.personalId = pers.id
WHERE NOT EXISTS (
    SELECT *
    FROM Note n
    WHERE c.id = n.clientId AND n.note = 'Cancelled'
);
"""
cursor.execute(query) 

#list of tuples where index 0 is number, index 1 is name, index 2 is program Name
client_programs = cursor.fetchall()

def fetch_user(pr_name, first_name):
        spread=Service_Account.open(pr_name)
        #retrieve client diary using their program name
        user_info=spread.worksheet("Goals/Info")
        #create a file to store the contents of their goals 
        with open("/tmp/context.csv", "w", newline='') as f:
                writer=csv.writer(f)
                writer.writerows(user_info.get_all_values())

        #use unstuctured loader to initialize the data source for RAG 
        docs=UnstructuredCSVLoader('/tmp/context.csv').load()

        #viola we have our loader, now what is the best way to summarize the user content as context for 
        #another model to use it as the very specific nature of response 


        # Define summarization prompt, using Directional Stimulus Prompting 
        # which utilizes stimulus/hints  which are keywords in this case, relating to the client's ideal workout/lifestyle mantra to provide fine
        #question specific guidance to GPT-4 for generating a summary

        prompt_template = """Write a concise summary of the following client's workout goals based on the hint:
        "{text}"
        Hint: Goals; Change; Optimization; Improvement; Consistency; Frequency; Healthy; Unhealthy; Weight Loss; Fat Loss; Weight Gain; Muscle Gain; Diet; Eating; Sleep.
        CONCISE SUMMARY:"""
        prompt = PromptTemplate.from_template(prompt_template)
        llm_chain = LLMChain(llm=chat_llm, prompt=prompt)


        # Define StuffDocumentsChain
        summary_chain = StuffDocumentsChain(
        llm_chain=llm_chain, document_variable_name="text"
        )


        user_workout_context=summary_chain.run(docs)
        
        #This is the final LLMChain prompt template, which will serve as a system prompt and the purpose of our daily motivational messages 
        temp="""
        
        You are BolicBot, a super passionate and intelligent fitness coach and motivator, 
        who can capture the hearts and minds of your clients by helping them push out of their comfort zone. 
        Your sole purpose for existence is to drive people to be better versions of themselves.
        Using your client's information here:  {context} , and their name:{name}, send them a very short and concise text, 
        motivating them to push through their very specific problems. 
        
        """

        
        prompt=PromptTemplate(input_variables=["context", "name"], template=temp)
        prompt.format(context=user_workout_context, name=first_name)
        chain=LLMChain(llm=chat_llm, prompt=prompt)
        return chain.run({"context": user_workout_context, "name": first_name})
        

for each_client in client_programs:
    answer=fetch_user(each_client[2], each_client[1])
    client.messages.create(to=each_client[0], from_=os.getenv("BOLIC_NUMBER"), body=answer) 
    
        

cursor.close()
connection.close()



