#Make Sure You have Installed the langchain

#Sentiment Analysis Prompt:

from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

# Define the prompt template for sentiment analysis
sentiment_template = '''Analyze the sentiment of the following statement:\n{input_text}'''

sentiment_analysis_prompt = PromptTemplate(
    input_variables=["input_text"],
    template=sentiment_template
)

# Format the sentiment analysis prompt
sentiment_analysis_prompt.format(input_text="I am feeling happy and excited.")

# Initialize the language model and chain
llm = OpenAI(temperature=0.7)
sentiment_analysis_chain = LLMChain(llm=llm, prompt=sentiment_analysis_prompt)

# Run the sentiment analysis chain
sentiment_analysis_chain.run("I am feeling Sad and excited.")
