from dotenv import load_dotenv
import os

from wxai_langchain.llm import LangChainInterface
from wxai_langchain.credentials import Credentials

from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods

from langchain.prompts import PromptTemplate

from datasets import load_dataset
huggingface_dataset_name = "knkarthick/dialogsum"
dataset = load_dataset(huggingface_dataset_name)

load_dotenv()

api_endpoint = 'https://us-south.ml.cloud.ibm.com'
api_key = os.getenv('API_KEY')
project_id = os.getenv('PROJECT_ID')

creds = Credentials(
    api_key=api_key,
    api_endpoint=api_endpoint,
    project_id=project_id
)

parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.SAMPLE,
    GenParams.MAX_NEW_TOKENS: 1024,
    GenParams.TEMPERATURE: 0,
    GenParams.TOP_K: 50,
    GenParams.TOP_P: 0.3,
}

llm = LangChainInterface(
    model=ModelTypes.FLAN_T5_XXL.value,
    params=parameters,
    credentials=creds
)

#### ZERO-SHOT PROMPTING ####

template = """
Summarize the following conversation
{dialogue}
Summary:
"""

example_indices = [40, 200]
dash_line = '-'.join('' for x in range(100))

prompt_template = PromptTemplate(template=template, input_variables=["dialogue"])
for i, index in enumerate(example_indices):
    dialogue = dataset['test'][index]['dialogue']
    summary = dataset['test'][index]['summary']
    
    prompt = prompt_template.format(dialogue=dialogue)
    output = llm(prompt)
    print(dash_line)
    print('Example ', i + 1)
    print(dash_line)
    print(f'INPUT PROMPT:\n{prompt}')
    print(dash_line)
    print(f'BASELINE HUMAN SUMMARY:\n{summary}')
    print(dash_line)    
    print(f'MODEL GENERATION - ZERO SHOT:\n{output}\n')

##### Example output:

# ---------------------------------------------------------------------------------------------------
# Example  1
# ---------------------------------------------------------------------------------------------------
# INPUT PROMPT:

# Summarize the following conversation
# #Person1#: What time is it, Tom?
# #Person2#: Just a minute. It's ten to nine by my watch.
# #Person1#: Is it? I had no idea it was so late. I must be off now.
# #Person2#: What's the hurry?
# #Person1#: I must catch the nine-thirty train.
# #Person2#: You've plenty of time yet. The railway station is very close. It won't take more than twenty minutes to get there.
# Summary:

# ---------------------------------------------------------------------------------------------------
# BASELINE HUMAN SUMMARY:
# #Person1# is in a hurry to catch a train. Tom tells #Person1# there is plenty of time.
# ---------------------------------------------------------------------------------------------------
# MODEL GENERATION - ZERO SHOT:
# Person1 must catch the 9: 30 train.

# ---------------------------------------------------------------------------------------------------
# Example  2
# ---------------------------------------------------------------------------------------------------
# INPUT PROMPT:

# Summarize the following conversation
# #Person1#: Have you considered upgrading your system?
# #Person2#: Yes, but I'm not sure what exactly I would need.
# #Person1#: You could consider adding a painting program to your software. It would allow you to make up your own flyers and banners for advertising.
# #Person2#: That would be a definite bonus.
# #Person1#: You might also want to upgrade your hardware because it is pretty outdated now.
# #Person2#: How can we do that?
# #Person1#: You'd probably need a faster processor, to begin with. And you also need a more powerful hard disc, more memory and a faster modem. Do you have a CD-ROM drive?
# #Person2#: No.
# #Person1#: Then you might want to add a CD-ROM drive too, because most new software programs are coming out on Cds.
# #Person2#: That sounds great. Thanks.
# Summary:

# ---------------------------------------------------------------------------------------------------
# BASELINE HUMAN SUMMARY:
# #Person1# teaches #Person2# how to upgrade software and hardware in #Person2#'s system.
# ---------------------------------------------------------------------------------------------------
# MODEL GENERATION - ZERO SHOT:
# Person2 wants to upgrade his system. Person1 suggests adding a painting program to his software. Person1 also suggests upgrading his hardware.