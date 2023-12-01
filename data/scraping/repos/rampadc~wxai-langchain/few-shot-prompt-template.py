from dotenv import load_dotenv
import os

from wxai_langchain.llm import LangChainInterface
from wxai_langchain.credentials import Credentials

from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods

from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate

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

#### FEW-SHOT PROMPTING ####
dash_line = '-'.join('' for x in range(100))

example_indices_full = [42, 50, 201, 300]
example_index_to_summarize = [200]

few_shot_examples = [{'dialogue': dataset['test'][i]['dialogue'], 'summary': dataset['test'][i]['summary']} for i in example_indices_full]
test_examples = [{'dialogue': dataset['test'][i]['dialogue'], 'summary': dataset['test'][i]['summary']} for i in example_index_to_summarize]

template = """
Dialogue:

{dialogue}

What was going on?

{summary}

"""
one_shot_template = PromptTemplate(
    input_variables=["dialogue", "summary"],
    template=template)
print(one_shot_template.format(**few_shot_examples[0]))


few_shot_template = FewShotPromptTemplate(
    examples=few_shot_examples,
    example_prompt=one_shot_template,
    suffix="Dialogue:\n\n{dialogue}\n\nWhat was going on?\n\n",
    input_variables=["dialogue"]
)

prompt = few_shot_template.format(dialogue=test_examples[0]["dialogue"])
output = llm(prompt)

print(dash_line)
print(prompt)
print(dash_line)
print(f'BASELINE HUMAN SUMMARY:\n{test_examples[0]["summary"]}\n')
print(dash_line)
print(f'MODEL GENERATION - FEW SHOTS:\n{output}')

#### Example output:
#
# Dialogue:

# #Person1#: I don't know how to adjust my life. Would you give me a piece of advice?
# #Person2#: You look a bit pale, don't you?
# #Person1#: Yes, I can't sleep well every night.
# #Person2#: You should get plenty of sleep.
# #Person1#: I drink a lot of wine.
# #Person2#: If I were you, I wouldn't drink too much.
# #Person1#: I often feel so tired.
# #Person2#: You better do some exercise every morning.
# #Person1#: I sometimes find the shadow of death in front of me.
# #Person2#: Why do you worry about your future? You're very young, and you'll make great contribution to the world. I hope you take my advice.

# What was going on?

# #Person1# wants to adjust #Person1#'s life and #Person2# suggests #Person1# be positive and stay healthy.


# ---------------------------------------------------------------------------------------------------

# Dialogue:

# #Person1#: I don't know how to adjust my life. Would you give me a piece of advice?
# #Person2#: You look a bit pale, don't you?
# #Person1#: Yes, I can't sleep well every night.
# #Person2#: You should get plenty of sleep.
# #Person1#: I drink a lot of wine.
# #Person2#: If I were you, I wouldn't drink too much.
# #Person1#: I often feel so tired.
# #Person2#: You better do some exercise every morning.
# #Person1#: I sometimes find the shadow of death in front of me.
# #Person2#: Why do you worry about your future? You're very young, and you'll make great contribution to the world. I hope you take my advice.

# What was going on?

# #Person1# wants to adjust #Person1#'s life and #Person2# suggests #Person1# be positive and stay healthy.




# Dialogue:

# #Person1#: Yeah. Just pull on this strip. Then peel off the back.
# #Person2#: You might make a few enemies this way.
# #Person1#: If they don't think this is fun, they're not meant to be our friends.
# #Person2#: You mean your friends. I think it's cruel.
# #Person1#: Yeah. But it's fun. Look at those two ugly old ladies. . . or are they men?
# #Person2#: Hurry! Get a shot!. . . Hand it over!
# #Person1#: I knew you'd come around. . .

# What was going on?

# #Person1# is about to make a prank. #Person2# thinks it's cruel at first but then joins.




# Dialogue:

# #Person1#: Where to, miss?
# #Person2#: Hi! Crenshaw and Hawthorne, at the Holiday Inn that is on that corner.
# #Person1#: Sure thing. So, where are you flying in from?
# #Person2#: From China.
# #Person1#: Really? You don't look very Chinese to me, if you don't mind me saying so.
# #Person2#: It's fine. I am actually from Mexico. I was in China on a business trip, visiting some local companies that manufacture bathroom products.
# #Person1#: Wow sounds interesting! Excuse me if I am being a bit nosy but, how old are you?
# #Person2#: Don't you know it's rude to ask a lady her age?
# #Person1#: Don't get me wrong! It's just that you seem so young and already doing business overseas!
# #Person2#: Well thank you! In that case, I am 26 years old, and what about yourself?
# #Person1#: I am 40 years old and was born and raised here in the good old U. S of A, although I have some Colombian heritage.
# #Person2#: Really? That's great! Do you speak some Spanish?
# #Person1#: Uh. . . yeah. . of course!
# #Person2#: Que bien! Sentences poems habeas en espanol!

# What was going on?

# #Person1# is driving #Person2# to an inn. They talk about their careers, ages, and where they was born.




# Dialogue:

# #Person1#: I cannot imagine if Trump were to be our President again.
# #Person2#: I am proud to say that he is our President, and I will be really happy if he could be re-elected.
# #Person1#: You voted for him, right?
# #Person2#: Did you vote for him, because I know that I did.
# #Person1#: I am not sure about this.
# #Person2#: I have nothing but faith in Trump.
# #Person1#: What?
# #Person2#: I am pretty sure he will make America great again!
# #Person1#: Well, though we do need some change in this country, I don't think he is the right person.
# #Person2#: Our country is already changing as it is.
# #Person1#: You are right about this.
# #Person2#: I trust that he will take good care of our country.
# #Person1#: Well, I don't think so. I will vote for Biden anyway.

# What was going on?

# #Person1# is crazy for Trump and voted for him. #Person2# doesn't agree with #Person1# on Trump and will vote for Biden.



# Dialogue:

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

# What was going on?


# ---------------------------------------------------------------------------------------------------
# BASELINE HUMAN SUMMARY:
# #Person1# teaches #Person2# how to upgrade software and hardware in #Person2#'s system.

# ---------------------------------------------------------------------------------------------------
# MODEL GENERATION - FEW SHOTS:
# Person2# needs to upgrade his system. He needs to add a painting program to his software. He also needs to upgrade his hardware.