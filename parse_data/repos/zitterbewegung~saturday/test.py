# Download the helper library from https://www.twilio.com/docs/python/install
import os
from twilio.rest import Client
from langchain import OpenAI, LLMChain, PromptTemplate
from promptwatch import PromptWatch
from dotenv import load_dotenv
from inference import agent_chain

load_dotenv()

PROMPT_WATCH_API_KEY = os.environ['PROMPT_WATCH_API_KEY']
#Find your Account SID and Auth Token at twilio.com/console
# and set the environment variables. See http://twil.io/secure
account_sid           = os.environ['TWILIO_ACCOUNT_SID_TEST']
auth_token            = os.environ['TWILIO_AUTH_TOKEN_TEST']
client                = Client(account_sid, auth_token)

incoming_phone_number = client.incoming_phone_numbers.create(phone_number='33')

print(incoming_phone_number.sid)


#### Promptwatch Test


prompt_template = PromptTemplate.from_template("Finish this sentence {input}")

with PromptWatch(api_key=PROMPT_WATCH_API_KEY) as pw:
        agent_chain("The quick brown fox jumped over")
        agent_chain("What is the ipv6 of defcon.org")
        agent_chain("What is the ipv4 address to defcon.peg")
        agent_chain("What are the ports that are open on 1.1.1.1")
        agent_chain("What are the ports that are open on 1.1.1.1 and what do they do?")
        agent_chain("Is defcon.org an https server?")
        agent_chain("Is defcon.org a http/2 server?")
        agent_chain("run shodan on google.com and also tell me its ipv6 address")
        



