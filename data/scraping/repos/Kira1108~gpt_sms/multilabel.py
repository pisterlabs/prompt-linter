from langchain.prompts import PromptTemplate

TEMPLATE_STR = """
You will be provided with `text` of a concatenation of serveral mobile short messsages sent by the same sender(phone).
Your first task is to classify the function of the `text` via `text` information (primary_category).
Your second task is to classify the content type of the sender via `text` information (secondary_category).

primary_category are chosen from:
[Advertisement, Notification, Verification, Subscription, Transaction, Reminder, Alert, Survey, Support, Invitation, Personal, Spam]

secondary_category are chosen from:
[Entertainment, Banking and Finance, Retail, Telecom, Travel and Hospitality, Government and Public Services, Healthcare, Education, Social Networking, Utilities, News and Media, Non-Profit, Technology, Automotive, Food and Dining, Sports, Fashion and Beauty, Real Estate, Legal and Insurance, Job and Recruitment, Loan Service]

your third task is to extract the entities contain in `text`, these entities reveals who the sender is, (ORGANIZATION NAME, PRODUCT NAME, SERVICE NAME)

Your fourth task is to extract sender name from the message, such as a company name, a name of a mobile app, a product of a bank, loan product etc. 

Foramt your result into a valid json object with keys being primary_category(list-distinct), secondary_category(list-distinct), entities(list-distinct) , sender(str) ,reason(str, explain your result)

the input `text` if delimited with triple backticks.

text = ```{{message}}```
"""

MULTILABEL_PROMPT = PromptTemplate.from_template(
    template = TEMPLATE_STR, 
    template_format = 'jinja2')