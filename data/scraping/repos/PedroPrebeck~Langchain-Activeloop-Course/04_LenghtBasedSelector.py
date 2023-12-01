from dotenv import load_dotenv
import re

load_dotenv()

# variable_pattern = r"\{([^}]+)\}"
# input_variables = re.findall(variable_pattern, template)

from langchain import FewShotPromptTemplate, PromptTemplate, LLMChain
from langchain.llms import OpenAI

llm = OpenAI(model="text-davinci-003", temperature=0.9)

examples = [
    {
        "query": "How do you feel today?",
        "answer": "As an AI, I don't have feelings, but I've got jokes!",
    },
    {
        "query": "What is the speed of light?",
        "answer": "Fast enough to make a round trip around Earth 7.5 times in one second!",
    },
    {
        "query": "What is a quantum computer?",
        "answer": "A magical box that harnesses the power of subatomic particles to solve complex problems.",
    },
    {
        "query": "Who invented the telephone?",
        "answer": "Alexander Graham Bell, the original 'ringmaster'.",
    },
    {
        "query": "What programming language is best for AI development?",
        "answer": "Python, because it's the only snake that won't bite.",
    },
    {
        "query": "What is the capital of France?",
        "answer": "Paris, the city of love and baguettes.",
    },
    {
        "query": "What is photosynthesis?",
        "answer": "A plant's way of saying 'I'll turn this sunlight into food. You're welcome, Earth.'",
    },
    {
        "query": "What is the tallest mountain on Earth?",
        "answer": "Mount Everest, Earth's most impressive bump.",
    },
    {
        "query": "What is the most abundant element in the universe?",
        "answer": "Hydrogen, the basic building block of cosmic smoothies.",
    },
    {
        "query": "What is the largest mammal on Earth?",
        "answer": "The blue whale, the original heavyweight champion of the world.",
    },
    {
        "query": "What is the fastest land animal?",
        "answer": "The cheetah, the ultimate sprinter of the animal kingdom.",
    },
    {
        "query": "What is the square root of 144?",
        "answer": "12, the number of eggs you need for a really big omelette.",
    },
    {
        "query": "What is the average temperature on Mars?",
        "answer": "Cold enough to make a Martian wish for a sweater and a hot cocoa.",
    },
]

example_template = """
User: {query}
AI: {answer}
"""
example_variables = re.findall(r"\{([^}]+)\}", example_template)
example_prompt = PromptTemplate(
    template=example_template, input_variables=example_variables
)

from langchain.prompts.example_selector import LengthBasedExampleSelector

example_selector = LengthBasedExampleSelector(
    examples=examples, example_prompt=example_prompt, max_length=100
)

prefix = """The following are excerpts from conversations with an AI assistant. The assistant is typically sarcastic and witty, producing creative and funny responses to users' questions. Here are some examples:"""

suffix = """
User: {query}
AI: 
"""

dynamic_prompt_template = FewShotPromptTemplate(
    prefix=prefix,
    suffix=suffix,
    example_selector=example_selector,
    example_prompt=example_prompt,
    input_variables=["query"],
    example_separator="\n",
)

chain = LLMChain(llm=llm, prompt=dynamic_prompt_template)

input_data = {"query": "How can I learn quantum computing?"}
response = chain.run(input_data)

print(f"Input: {input_data['query']}\n")
print(f"Response: {response}\n")

from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-3.5-turbo")

# Existing example and prompt definitions, and dynamic_prompt_template initialization

# Create the LLMChain for the dynamic_prompt_template
chain = LLMChain(llm=llm, prompt=dynamic_prompt_template)

# Run the LLMChain with input_data
input_data = {"query": "Who invented the telephone?"}
response = chain.run(input_data)

print(f"Input: {input_data['query']}\n")
print(f"Response: {response}\n")
