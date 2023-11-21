from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

# Define the prompt template for meeting scheduling
meeting_template = '''Schedule a meeting with the following details:
Date: {date}
Time: {time}
Participants: {participants}
Agenda: {agenda}'''

meeting_prompt = PromptTemplate(
    input_variables=["date", "time", "participants", "agenda"],
    template=meeting_template
)

# Format the meeting scheduling prompt
meeting_prompt.format(
    date="June 30, 2023",
    time="2:00 PM",
    participants="John, Mary, David",
    agenda="Discuss project updates"
)

# Initialize the language model and chain
llm = OpenAI(temperature=0.7)
meeting_chain = LLMChain(llm=llm, prompt=meeting_prompt)

# Run the meeting scheduling chain
meeting_chain.run({
    "date": "June 30, 2023",
    "time": "2:00 PM",
    "participants": "John, Mary, David",
    "agenda": "Discuss project updates"
})




#OUTPUT

"""


Dear John, Mary and David,

I would like to invite you to a meeting on June 30, 2023 at 2:00 PM to discuss project updates.

Please let me know if you are able to attend.

Thank you,
[Your Name]

"""
