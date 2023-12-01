from langchain import LLMChain
from langchain.chains import SimpleSequentialChain, SequentialChain, TransformChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import os
import openai
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = "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"
api_key = "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"
# llm = OpenAI()
llm = ChatOpenAI(openai_api_key=api_key, temperature=0)
embedding_function = OpenAIEmbeddings()


yelp_review = open('yelp_review.txt').read()
print(yelp_review)

# text = yelp_review['text']
# only_review_text = text.split('REVIEW:')[-1]
# lower_case_text = only_review_text.lower()

def transformer_fun(inputs: dict) -> dict:
    text = inputs['text']
    only_review_text = text.split('REVIEW:')[-1]
    lower_case_text = only_review_text.lower()
    return {'output': lower_case_text}


transform_chain = TransformChain(input_variables=['text'],
                                 output_variables=['output'],
                                 transform=transformer_fun)

template = "Create a one sentence summary of this review:\n{review_text}"

prompt = ChatPromptTemplate.from_template(template)
summary_chain = LLMChain(llm=llm,
                     prompt=prompt,
                     output_key="review_summary")

sequential_chain = SimpleSequentialChain(chains=[transform_chain,summary_chain],
                                        verbose=True)
result = sequential_chain(yelp_review)
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2")
print(result['output'])
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2")
print(result['input'])