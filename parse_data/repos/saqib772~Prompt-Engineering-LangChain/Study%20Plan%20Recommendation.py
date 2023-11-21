from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

# Define the prompt template for study plan recommendation
study_plan_template = '''Create a personalized study plan based on the following information:
Subject: {subject}
Study Duration: {duration}
Learning Style: {learning_style}'''

study_plan_prompt = PromptTemplate(
    input_variables=["subject", "duration", "learning_style"],
    template=study_plan_template
)

# Format the study plan recommendation prompt
study_plan_prompt.format(
    subject="Mathematics",
    duration="3 months",
    learning_style="Visual"
)

# Initialize the language model and chain
llm = OpenAI(temperature=0.7)
study_plan_chain = LLMChain(llm=llm, prompt=study_plan_prompt)

# Run the study plan recommendation chain
study_plan_chain.run({
    "subject": "Mathematics",
    "duration": "3 months",
    "learning_style": "Visual"
})
