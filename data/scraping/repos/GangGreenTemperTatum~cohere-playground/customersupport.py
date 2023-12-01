import os
import re
import cohere 

api_key = os.environ["CO_KEY"]
co = cohere.Client(api_key)

from cohere.responses.classify import Example

examples=[
  Example("How do I find my insurance policy?", "Finding policy details"),
  Example("How do I download a copy of my insurance policy?", "Finding policy details"),
  Example("How do I find my policy effective date?", "Finding policy details"),
  Example("When does my insurance policy end?", "Finding policy details"),
  Example("Could you please tell me the date my policy becomes effective?", "Finding policy details"),
  Example("How do I sign up for electronic filing?", "Change account settings"),
  Example("How do I change my policy?", "Change account settings"),
  Example("How do I sign up for direct deposit?", "Change account settings"),
  Example("I want direct deposit. Can you help with that?", "Change account settings"),
  Example("Could you deposit money into my account rather than mailing me a physical cheque?", "Change account settings"),
  Example("How do I file an insurance claim?", "Filing a claim and viewing status"),
  Example("How do I file a reimbursement claim?", "Filing a claim and viewing status"),
  Example("How do I check my claim status?", "Filing a claim and viewing status"),
  Example("When will my claim be reimbursed?", "Filing a claim and viewing status"),
  Example("I filed my claim 2 weeks ago but I still haven’t received a deposit for it.", "Filing a claim and viewing status"),
  Example("I want to cancel my policy immediately! This is nonsense.", "Cancelling coverage"),
  Example("Could you please help my end my insurance coverage? Thank you.",
  "Cancelling coverage"),
  Example("Your service sucks. I’m switching providers. Cancel my coverage.", "Cancelling coverage"),
  Example("Hello there! How do I cancel my coverage?", "Cancelling coverage"),
  Example("How do I delete my account?", "Cancelling coverage")
]

inputs=[" I want to change my password",  
        "Does my policy cover prescription medication",  
        ]

response = co.classify(  
    model='large',  
    inputs=inputs,  
    examples=examples)

print(response.classifications)