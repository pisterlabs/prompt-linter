from langchain.utilities import GoogleSearchAPIWrapper
from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.document_loaders import WikipediaLoader
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import xml.etree.ElementTree as ET
import ast
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import time
import re
import json
import anthropic

load_dotenv()

class ClaimExtractor:

# {
#   "paragraph": "Thank you Elon. Look, let me be clear, Putin’s invasion of Ukraine is an unprovoked act of aggression that threatens democracy in Europe and beyond. My administration has led the effort to support the brave Ukrainian people with military aid, humanitarian relief and crippling sanctions on Russia. But there are still tough days ahead. We’ll continue rallying our NATO allies to stand united against Russian aggression. And I was always supplying Ukraine’s fighters with the weapons they needed to defend their homeland, even as Putin continued his brutal assaults on civilians. We’ll welcome Ukrainian refugees with open arms. Most importantly, we’ll keep standing on the side of freedom and sovereignty. Putin wants to destroy the international order. On my watch, that simply won’t happen. The free world will meet this test - democracy will prevail over tyranny.",
#   "speaker": "Donald Trump",
#   "opponent": "Joe Biden",
#   "moderator": "Elon Musk"
# }

    def __init__(self, client):
        self.anthropic = client

    def extractClaims(self, paragraph):
        claim_extraction_prompt = """
this is a paragraph out of a debate between """+ paragraph["speaker"] +""" and """+ paragraph["opponent"] +""", moderated by """+ paragraph["moderator"] +""". """+ paragraph["speaker"] +""" says:
<paragraph>"""+ paragraph["paragraph"] +"""</paragraph>
<instruction>

Provide me with a list of strings in python of all discrete statements about the past, or accusatory statements about something their opponent did, that were claimed in the provided paragraph stated by """+ paragraph["speaker"] +""".
Forward-looking statements, opinions, etc. are strongly prohibited!

For each claim that you add to the list, make sure that you provide enough context so that a fact-checker can understand the claim and its context without any additional information.

Do not give an explanation on your provided list, just provide the list of claims with additional context needed to understand it.
You can rephrase the claims if you think that it is necessary to provide enough context to understand the claim. You can also rephrase unspecific words (like "they", "she", "her", etc.) with who was actually meant by that.

Make sure, that the output is a python list of strings.

Remember: Forward-looking statements, opinions, etc. are strongly prohibited!

<response_information>
list: a python list of the extracted claims
</response_information>

</instruction>
<example_response><list>["There are no longer any old growth forests left in France.", "The climate of southern France can no longer support olive trees due to climate change.", "Invasive insect species have destroyed 30 percent of grapevines in France."]</list></example_response>"""

# Forward-looking statements, opinions, etc. should not be included.

# Provide me with a python list of all the claims from the paragraph stated by """+ paragraph["speaker"] +""" which need to be fact-checked.
# Only provide claims that are targeting past events, as we cannot fact-check something that will happen in the future.
# For each claim that you add to the list, make sure that you provide enough context so that a fact-checker can understand the claim and its context without any additional information.


# Extract all claims from the paragraph stated by """+ paragraph["speaker"] +""" which need to be fact-checked.
# In the best case you can provide the exact wording of the claim, but if you are unsure, you can also provide a paraphrased version of the claim that better describes the context of the claim. 
# Make sure that you provide enough context to each claim, so that a fact-checker can later understand the claim and the context of the claim.
# Make sure that you include all the claims that need to be fact-checked. Do not forget any claims!

        claims = self.anthropicGetList(claim_extraction_prompt)
        
        claims_output = []
        for claim in claims:
            claims_output.append({"claim": claim, "speaker": paragraph["speaker"], "opponent": paragraph["opponent"]})

        return claims_output
    
    def anthropicGetList(self, prompt):
        successful = False
        while not successful:
            try:
                completion = self.anthropic.completions.create(
                    model="claude-2",
                    max_tokens_to_sample=10000,
                    temperature=0.0,
                    prompt=f"{HUMAN_PROMPT}{prompt}{AI_PROMPT}<list>")
                # print(completion.completion)
                result = completion.completion
                successful = True
            except:
                print("Anthropic error: Trying again...")
                time.sleep(3)
        
        try:
            result_list = ast.literal_eval(result.replace("</list>",""))
        except:
            print("----------------------------------------------------")
            print("Error: ", result)
            result_list = []

        print(result_list)
        return result_list


# speaker = "Donald Trump"
# opponent = "Joe Biden"
# moderator = "Elon Musk"
# paragraph = "Thank you Elon. Look, let me be clear, Putin’s invasion of Ukraine is an unprovoked act of aggression that threatens democracy in Europe and beyond. My administration has led the effort to support the brave Ukrainian people with military aid, humanitarian relief and crippling sanctions on Russia. But there are still tough days ahead. We’ll continue rallying our NATO allies to stand united against Russian aggression. And I was always supplying Ukraine’s fighters with the weapons they needed to defend their homeland, even as Putin continued his brutal assaults on civilians. We’ll welcome Ukrainian refugees with open arms. Most importantly, we’ll keep standing on the side of freedom and sovereignty. Putin wants to destroy the international order. On my watch, that simply won’t happen. The free world will meet this test - democracy will prevail over tyranny. We owned the western part of Germany back then after the 2nd world war, you know! I was speaking to Kim Jong Un in 2019!"
# paragraph = "You either do, Chris, a solicited ballot, where you’re sending it in, they’re sending it back and you’re sending. They have mailmen with lots of it. Did you see what’s going on? Take a look at West Virginia, mailman selling the ballots. They’re being sold. They’re being dumped in rivers. This is a horrible thing for our country."
# paragraph = "We're going to invest in American workers and American ingenuity. We need to build infrastructure like roads, bridges, universal broadband. We'll lower taxes for the middle class and make corporations pay their fair share. And I'll fight against outsourcing and make sure more products are stamped Made in America."

# client = anthropic.Client(api_key=os.getenv("ANTHROPIC_API_KEY"))
# claim_extractor = ClaimExtractor(client)
# paragraph_dict = dict()
# paragraph_dict["speaker"] = speaker
# paragraph_dict["opponent"] = opponent
# paragraph_dict["moderator"] = moderator
# paragraph_dict["paragraph"] = paragraph

# claims = claim_extractor.extractClaims(paragraph_dict)
# print(claims)

# truthGPT = FactChecker(client)

# for claim in claims:
#     fact_checking_result = truthGPT.factCheck(claim)
#     print(fact_checking_result)

