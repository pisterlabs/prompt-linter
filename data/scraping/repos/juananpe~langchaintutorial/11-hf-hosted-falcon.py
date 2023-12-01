from langchain.llms import HuggingFaceEndpoint
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain

''' Based on: https://huggingface.co/tiiuae/falcon-40b-instruct/discussions/51 '''

HUGGINGFACEHUB_API_TOKEN = ""
from dotenv import load_dotenv
load_dotenv()

endpoint_url = (
            'https://YOUR_ENDPOINT.us-east-1.aws.endpoints.huggingface.cloud'
)
hf = HuggingFaceEndpoint(
    endpoint_url=endpoint_url,
    huggingfacehub_api_token= HUGGINGFACEHUB_API_TOKEN,
    task = 'text-generation',
    model_kwargs = {
                "min_length": 200,
                "max_length":2000,
                "temperature":0.01,
                "max_new_tokens":200,
                "num_return_sequences":1
            }
)

template = """Context: {context}
Question: {question}
Answer: """

prompt = PromptTemplate(template=template, input_variables=["question", "context"])
llm_chain = LLMChain(prompt=prompt, llm=hf, verbose=True)

context = '''Don't make up your response. If you don't know it, just tell me you don't know.
'''
question = "What's the difference between fusion and fission?"

print(llm_chain.run({'question': question, 'context': context}))