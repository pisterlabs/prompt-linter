from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')
import os

from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import DeepLake
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate, FewShotPromptTemplate

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="""
Input: {input}\nOutput: {output}
"""
)

examples = [
    {"input": "0°C", "output": "32°F"},
    {"input": "10°C", "output": "50°F"},
    {"input": "20°C", "output": "68°F"},
    {"input": "30°C", "output": "86°F"},
    {"input": "40°C", "output": "104°F"},
]

activeloop_dataset_name = "langchain_course_fewshot_selector"
dataset_path = f"hub://{os.environ.get('ACTIVELOOP_ORGID')}/{activeloop_dataset_name}"
vecdb = DeepLake(dataset_path=dataset_path)

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    embeddings,
    vecdb,
    k=1,
)

similar_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Convert the following temperature from Celsius to Fahrenheit:\n",
    suffix="\nInput: {temperature}\nOutput:",
    input_variables=["temperature"],
)


print(similar_prompt.format(temperature="25°C"))
print(similar_prompt.format(temperature="35°C"))

similar_prompt.example_selector.add_example({"input": "50°C", "output": "122°F"})
print(similar_prompt.format(temperature="40°C"))