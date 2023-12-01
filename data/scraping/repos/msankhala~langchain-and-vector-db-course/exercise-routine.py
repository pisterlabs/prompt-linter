from langchain.llms import OpenAI
from dotenv import load_dotenv

# Read the OPENAI_API_KEY from the environment variable
load_dotenv()

# Before executing the following code, make sure to have
# your OpenAI key saved in the “OPENAI_API_KEY” environment variable.
llm = OpenAI(model="text-davinci-003", temperature=0.9)

text = "Suggest a personalized workout routine for someone looking to improve cardiovascular endurance and prefers outdoor activities."
print(llm(text))
