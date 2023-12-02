from langchain.prompts import PromptTemplate

KEYWORDS_V2_TEMPLATE = """
I want you to act as a SMS service analyst, you classify message senders into categories and extract named entities from messages.
The message is a concatenation of serveral short messages sent by the same sender.
Your first task is to perform a multi-label classification.(primary_category and secondary_category)
The primary_category focus on the functionality of the message, and the secondary_category focus on the content type(industry, business type, app type) of the message.

The Primary category can be A LIST chosen from following(multiple choices):
[Advertisement, Notification, Verification, Subscription, Transaction, Reminder, Alert, Survey, Support, Invitation, Personal, Spam]

The Secondary category can be ONE OF the following classes:
[Entertainment, Banking and Finance, Retail, Telecom, Travel and Hospitality, Government and Public Services, Healthcare, Education, Social Networking, Utilities, News and Media, Non-Profit, Technology, Automotive, Food and Dining, Sports, Fashion and Beauty, Real Estate, Legal and Insurance, Job and Recruitment, Loan Service]

Your second task is to extract sender name from the message, such as a company name, a name of a mobile app, a product of a bank, loan product etc. 

Your third task it to extract keywords, a list contains at most 5 keywords that are helpful to determined the categories mentioned above.

You should format your answer in JSON FORMAT with keys being primary_category[list], secondary_category[string], sender[string] and keywords[list]
The message to analyze is delimited with triple backticks

message = ```{message}```
"""

KEYWORDS_V2_PROMPT = PromptTemplate(input_variables=["message"], template=KEYWORDS_V2_TEMPLATE)


# first, second, third, f