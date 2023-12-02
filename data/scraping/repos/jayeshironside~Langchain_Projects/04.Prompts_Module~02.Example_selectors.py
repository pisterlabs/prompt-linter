import os
from dotenv import load_dotenv
# Use the environment variables to retrieve API keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from langchain.llms import OpenAI

# Few Shot Templates
#Few-shot learning is a way to teach computers to make predictions using only a small amount of information. Instead of needing lots of examples, computers can learn from just a few examples. They find patterns in the examples and use those patterns to understand and recognize new things. It helps computers learn quickly and accurately with only a little bit of information.

our_prompt = """You are a 5 year old girl, who is very funny,mischievous and sweet: 

Question: What is a house?
Response: """

llm = OpenAI(temperature=.9, model="text-davinci-003")
print("Output 1 :-",llm(our_prompt))

# We observe that though we have instructed the model to act as a little girl, it's unable to do so as it very generic by nature So we will try to proved some external knowledge to get the perfect answers from it

our_prompt = """You are a 5 year old girl, who is very funny,mischievous and sweet: 
Here are some examples: 

Question: What is a mobile?
Response: A mobile is a magical device that fits in your pocket, like a mini-enchanted playground. It has games, videos, and talking pictures, but be careful, it can turn grown-ups into screen-time monsters too!

Question: What are your dreams?
Response: My dreams are like colorful adventures, where I become a superhero and save the day! I dream of giggles, ice cream parties, and having a pet dragon named Sparkles..

Question: What is a house?
Response: """

print("Output 2 :-",llm(our_prompt))

# The FewShotPromptTemplate feature offered by LangChain allows for few-shot learning using prompts. In the context of large language models (LLMs), the primary sources of knowledge are parametric knowledge (learned during model training) and source knowledge (provided within model input at inference time). The FewShotPromptTemplate enables the inclusion of a few examples within prompts, which the model can read and use to apply to user input, enhancing the model's ability to handle specific tasks or scenarios.
from langchain.prompts import PromptTemplate
from langchain import FewShotPromptTemplate

examples = [
    {
        "query": "What is a mobile?",
        "answer": "A mobile is a magical device that fits in your pocket, like a mini-enchanted playground. It has games, videos, and talking pictures, but be careful, it can turn grown-ups into screen-time monsters too!"
    }, {
        "query": "What are your dreams?",
        "answer": "My dreams are like colorful adventures, where I become a superhero and save the day! I dream of giggles, ice cream parties, and having a pet dragon named Sparkles.."
    }
]

# Lets create a sample template
example_template = """
Question: {query}
Response: {answer}
"""

example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

# The previous original prompt can be divided into a prefix and suffix. The prefix consists of the instructions or context given to the model, while the suffix includes the user input and output indicator.

prefix = """You are a 5 year old girl, who is very funny,mischievous and sweet: 
Here are some examples: 
"""

suffix = """
Question: {userInput}
Response: """

# Let's create a few shot prompt template, by using the above details
few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["userInput"],
    example_separator="\n\n"
)

query = "What is a house?"
print("Output 3 :-",llm(few_shot_prompt_template.format(userInput=query)))

# Adding more examples so that model can have more context before responding with a answer
examples = [
    {
        "query": "What is a mobile?",
        "answer": "A mobile is a magical device that fits in your pocket, like a mini-enchanted playground. It has games, videos, and talking pictures, but be careful, it can turn grown-ups into screen-time monsters too!"
    }, {
        "query": "What are your dreams?",
        "answer": "My dreams are like colorful adventures, where I become a superhero and save the day! I dream of giggles, ice cream parties, and having a pet dragon named Sparkles.."
    }, {
        "query": " What are your ambitions?",
        "answer": "I want to be a super funny comedian, spreading laughter everywhere I go! I also want to be a master cookie baker and a professional blanket fort builder. Being mischievous and sweet is just my bonus superpower!"
    }, {
        "query": "What happens when you get sick?",
        "answer": "When I get sick, it's like a sneaky monster visits. I feel tired, sniffly, and need lots of cuddles. But don't worry, with medicine, rest, and love, I bounce back to being a mischievous sweetheart!"
    }, {
        "query": "WHow much do you love your dad?",
        "answer": "Oh, I love my dad to the moon and back, with sprinkles and unicorns on top! He's my superhero, my partner in silly adventures, and the one who gives the best tickles and hugs!"
    }, {
        "query": "Tell me about your friend?",
        "answer": "My friend is like a sunshine rainbow! We laugh, play, and have magical parties together. They always listen, share their toys, and make me feel special. Friendship is the best adventure!"
    }, {
        "query": "What math means to you?",
        "answer": "Math is like a puzzle game, full of numbers and shapes. It helps me count my toys, build towers, and share treats equally. It's fun and makes my brain sparkle!"
    }, {
        "query": "What is your fear?",
        "answer": "Sometimes I'm scared of thunderstorms and monsters under my bed. But with my teddy bear by my side and lots of cuddles, I feel safe and brave again!"
    }
]

# In the above explanation, be have been using 'FewShotPromptTemplate' and 'examples' dictionary as it is more robust approach compared to using a single f-string. It offers features such as the ability to include or exclude examples based on the length of the query. This is important because there is a maximum context window limitation for prompt and generation output length. The goal is to provide as many examples as possible for few-shot learning without exceeding the context window or increasing processing times excessively. The dynamic inclusion/exclusion of examples means that we choose which examples to use based on certain rules. This helps us use the model's abilities in the best way possible. It allows us to be efficient and make the most out of the few-shot learning process.

from langchain.prompts.example_selector import LengthBasedExampleSelector

# LengthBasedExampleSelector - This ExampleSelector chooses examples based on length, useful to prevent prompt exceeding context window. It selects fewer examples for longer inputs and more for shorter ones, ensuring prompt fits within limits. The maximum length of the formatted examples is set to 'n' characters. To determine which examples to include, the length of a string is measured using the get_text_length function, which is provided as a default value if not specified.

example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=200
)

# Creating a new dynamic few shot prompt template And we are passing example_selector instead of examples as earlier
new_prompt_template = FewShotPromptTemplate(
    example_selector=example_selector,  # use example_selector instead of examples
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["userInput"],
    example_separator="\n"
)

query = "What is a house?"
print("Output 4 :-",llm(new_prompt_template.format(userInput=query)))

# We can also add an extra example to an example selector we already have.
new_example = {"query": "What's your favourite work?", "answer": "sleep"}
new_prompt_template.example_selector.add_example(new_example)

example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=1000
)

print(llm(new_prompt_template.format(userInput=query)))