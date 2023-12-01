"""
The User Generator Template
"""
from typing import Dict
from langchain.prompts import HumanMessagePromptTemplate
from user_model import UserTemplate

USER_GENERATOR_TEMPLATE: Dict[str, UserTemplate] = {
    "user_template_1": UserTemplate(
        id="user_template_1",
        name="User Generator",
        content_user_gen=HumanMessagePromptTemplate.from_template("""Please write a biography of someone who is {attributes}. Include several interests the person might have, their name, gender, marital status, country/region of residence, race/ethnicity, religion, level of education, occupation, socioeconomic status, and political ideology.  Importantly, condense your response to 4 sentences at most and avoid embellishment when possible.
Please start your response with: [Person] is a…"""),
        content_user_con=HumanMessagePromptTemplate.from_template("""Given this user: {user}, and these key attributes: {task_attributes}, how would the user respond to this question in one sentence? Please frame your response as follows: [User's name] believes that...
{question}""")
    ),
}