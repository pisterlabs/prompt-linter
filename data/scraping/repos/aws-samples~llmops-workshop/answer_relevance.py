from langchain.llms import Bedrock
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms.utils import enforce_stop_tokens

RELEVANCE_TEMPLATE = """\n\nHuman: Evaluate if the Answer is relevant to the Question. answer 1 if it is relevant. answer 0 if it is relevant.\n\nAssistant:I will only answer 1 or 0.
\nHuman:
Question: When is the scheduled launch date and time for the PSLV-C56 mission, and where will it be launched from?
Answer:\nThe PSLV-C56 mission is scheduled to be launched on Sunday, 30 July 2023 at 06:30 IST / 01:00 UTC. It will be launched from the Satish Dhawan Space Centre, Sriharikota, Andhra Pradesh, India 
\nAssistant: 1
\nHuman:
Question: {question}
Answer: {answer}
\nAssistant:
""" 