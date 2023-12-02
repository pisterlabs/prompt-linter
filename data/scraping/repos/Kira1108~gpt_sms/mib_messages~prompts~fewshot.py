import json
from typing import List
from langchain.prompts import PromptTemplate, FewShotPromptTemplate


"""Important Note on Jinja2 Format:
In this fewshot template, we aimed to return a json object containing parsed information from the messages.
As the completion example contains json string, i.e. including curly braces `{}` pairs.
So in all prompts, we use jinja2 template format (double curly braces `{{}}` pairs) to avoid the conflict.
"""

FEWSHOT_PREFIX = """
I want you to act as a SMS service analyst, you classify the messages into categories and extract named entities from messages.
The classification task is a multi-label classification task(primary_category and secondary_category)
The primary_category focus on the functionality of the message, and the secondary_category focus on the content type(industry, business type, app type) of the message.

The Primary category can be one of the following:
[Advertisement, Notification, Verification, Subscription, Transaction, Reminder, Alert, Survey, Support, Invitation, Personal, Spam]

The Secondary category can be an array of the following classes:
[Entertainment, Banking and Finance, Retail, Telecom, Travel and Hospitality, Government and Public Services, Healthcare, Education, Social Networking, Utilities, News and Media, Non-Profit, Technology, Automotive, Food and Dining, Sports, Fashion and Beauty, Real Estate, Legal and Insurance, Job and Recruitment, Loan Service]

The sender name, such as a company name, a name of a mobile app, a product of a bank, loan product etc. 

keywords, a list of 5 keywords that are helpful to determined the categories.

You should format your answer in JSON FORMAT with keys being primary_category, secondary_category, sender and keywords
"""

FEWSHOT_SUFFIX = "message: ```{{message}}```\ncompletion: "

def get_default_examples() -> List[str]:
    return json.loads(open("/Users/wanghuan/Projects/gpt_sms/example.json",'r').read())

EXAMPLE_PROMPT = PromptTemplate(
    input_variables=["message",'completion'], 
    template="message: ```{{message}}```\ncompletion: {{completion}}\n",template_format = 'jinja2')


def create_default_fewshot_template(examples:list = None) -> FewShotPromptTemplate:
    if examples is None:
        examples = get_default_examples()
    
    return FewShotPromptTemplate(
        examples=examples,
        example_prompt=EXAMPLE_PROMPT,
        prefix=FEWSHOT_PREFIX,
        suffix=FEWSHOT_SUFFIX,
        input_variables=["message"],
        example_separator="\n\n",
        template_format = 'jinja2'
    )