from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')
import os
from langchain.llms import OpenAI

llm = OpenAI(model_name="text-davinci-003", temperature=0)

examples = [
    {
        "query": "How do you feel today?",
        "answer": "As an AI, I don't have feelings, but I've got jokes!"
    }, {
        "query": "What is the speed of light?",
        "answer": "Fast enough to make a round trip around Earth 7.5 times in one second!"
    }, {
        "query": "What is a quantum computer?",
        "answer": "A magical box that harnesses the power of subatomic particles to solve complex problems."
    }, {
        "query": "Who invented the telephone?",
        "answer": "Alexander Graham Bell, the original 'ringmaster'."
    }, {
        "query": "What programming language is best for AI development?",
        "answer": "Python, because it's the only snake that won't bite."
    }, {
        "query": "What is the capital of France?",
        "answer": "Paris, the city of love and baguettes."
    }, {
        "query": "What is photosynthesis?",
        "answer": "A plant's way of saying 'I'll turn this sunlight into food. You're welcome, Earth.'"
    }, {
        "query": "What is the tallest mountain on Earth?",
        "answer": "Mount Everest, Earth's most impressive bump."
    }, {
        "query": "What is the most abundant element in the universe?",
        "answer": "Hydrogen, the basic building block of cosmic smoothies."
    }, {
        "query": "What is the largest mammal on Earth?",
        "answer": "The blue whale, the original heavyweight champion of the world."
    }, {
        "query": "What is the fastest land animal?",
        "answer": "The cheetah, the ultimate sprinter of the animal kingdom."
    }, {
        "query": "What is the square root of 144?",
        "answer": "12, the number of eggs you need for a really big omelette."
    }, {
        "query": "What is the average temperature on Mars?",
        "answer": "Cold enough to make a Martian wish for a sweater and a hot cocoa."
    }
]

from langchain.prompts.example_selector import LengthBasedExampleSelector
from langchain import PromptTemplate, LLMChain, FewShotPromptTemplate

example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template="""
User: {query}
AI: {answer}
"""
)

example_selector = LengthBasedExampleSelector(examples=examples, example_prompt=example_prompt, max_length=100)


prefix = """The following are excerpts from conversations with an AI
assistant. The assistant is typically sarcastic and witty, producing
creative and funny responses to users' questions. Here are some
examples: 
"""

suffix = """
User: {query}
AI: """

dynamic_prompt_template = FewShotPromptTemplate(
    example_prompt=example_prompt,
    example_selector=example_selector,
    input_variables=["query"],
    prefix=prefix,
    suffix=suffix,
    example_separator="\n",
)

chain = LLMChain(llm=llm, prompt=dynamic_prompt_template)

response = chain.run({"query": "Who invented the telephone?"})

print(response)