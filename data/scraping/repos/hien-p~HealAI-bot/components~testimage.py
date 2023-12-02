from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain import LLMChain
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv
from textblob import TextBlob

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")


# define the output
response_schemas = [
    ResponseSchema(name="sentiment", description="a sentiment label based on the user text. It should be either Negative, Positive or Neutral"),
    ResponseSchema(name="reason", description="""
    If the sentiment is Negative then return the reason why the user shouldn't have said that.
    If the sentiment is Positive then return a compliment.
    For Neutral then return a instruct for a better message. 
    """),
    ResponseSchema(name="reply", description="the best and friendliest replacement to the given user text")
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
# prompt template
template = """You are good at detecting human emotion. All emotions you know are Negative, Positive and Neutral.
Given a human text, subjectivity and polarity, your job is to answer as best as possible.
Know that subjectivity is a measure of how much of a text is based on personal opinions or beliefs, rather than on facts. 
It is a float value between 0 and 1, where 0 represents an objective text and 1 represents a highly subjective text.
Also know that polarity is a indicator for the sentiment of the given user text, negative value means Negative, positive value means Positive and 0 means Neutral.
{format_instructions}
User text: {text}
Subjectivity: {subjectivity}
Polarity: {polarity}"""



format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(template=template, input_variables=["text","subjectivity","polarity"],
                        partial_variables={"format_instructions": format_instructions})
model = OpenAI(verbose=True, temperature=0.0)
# Build chain
sentiment_chain = LLMChain(llm=model, prompt=prompt, output_key='result')

userText = "You are idiot"
tb = TextBlob(userText)
subjectivity = tb.subjectivity
polarity = round(tb.polarity, 2)


ans = sentiment_chain({"text": userText, "polarity": polarity, "subjectivity": subjectivity})
print(ans)






# test emotions

emotions = ['Happy 😊','Sad 😔','Angry 😠','Surprise 😲','Fear 😨']


for emo in emotions:
    emos = emo.split(" ")
    des = f"""a js object contains two properties.
    The first one is label: str // always return '{emos[1]}' """
    schema = ResponseSchema(name=emos[0], description=des)
    
# define the output
response_schemas = [
    ResponseSchema(name="sentiment", description="a sentiment label based on the user text. It should be either Negative, Positive or Neutral"),
    ResponseSchema(name="reason", description="the reason behind your answer"),
    ResponseSchema(name="reply", description="the best reply to the given user text")
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
# prompt template
template = """You are good at detecting human emotion. All emotions you know are Negative, Positive and Neutral.
Given a human text, subjectivity and polarity, your job is to answer as best as possible.
Know that subjectivity is a measure of how much of a text is based on personal opinions or beliefs, rather than on facts. 
It is a float value between 0 and 1, where 0 represents an objective text and 1 represents a highly subjective text.
Also know that polarity is a indicator for the sentiment of the given user text, negative value means Negative, positive value means Positive and 0 means Neutral.
{format_instructions}
User text: {text}
Subjectivity: {subjectivity}
Polarity: {polarity}"""

format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(template=template, input_variables=["text","subjectivity","polarity"],
                        partial_variables={"format_instructions": format_instructions})

# Build chain
sentiment_chain = LLMChain(llm=model, prompt=prompt, output_key='result')
# print(sentiment_chain.)

ans = sentiment_chain({"text": userText, "polarity": polarity, "subjectivity": subjectivity})