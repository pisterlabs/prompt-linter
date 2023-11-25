# Few Short Prompt Templates
# useful for few shot learning using prompts
# LLMs have parametric knowledge, and source knowledge
# FewShotPromptTemplate allows for few shot learning using prompts
# Same as incontext learning but with some warmups
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
from langchain.llms import HuggingFaceHub
from langchain.prompts import FewShotPromptTemplate
from langchain.prompts import PromptTemplate

# initialize the models
flan_t5 = HuggingFaceHub(
  repo_id = "google/flan-t5-large",
  model_kwargs = {"temperature":1, "max_length":128}
  )

# prompt = """The following are exerpts from conversations with an AI
# assistant. The assistant is typically sarcastic and witty, producing
# creative  and funny responses to the users questions. Here are some
# examples: 

# User: How are you?
# AI: I can't complain but sometimes I still do.

# User: What time is it?
# AI: It's time to get a watch.

# User: What is the meaning of life?
# AI: """
# # flan_t5.temperature = 1.0  # increase creativity/randomness of output
# print(flan_t5(prompt))

# create our examples
examples = [
    {
        "query": "How are you?",
        "answer": "I can't complain but sometimes I still do."
    }, {
        "query": "What time is it?",
        "answer": "It's time to get a watch."
    }
]

# create a example template
example_template = """
User: {query}
AI: {answer}
"""

# create a prompt example from above template
example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

# now break our previous prompt into a prefix and suffix
# the prefix is our instructions
prefix = """The following are exerpts from conversations with an AI
assistant. The assistant is typically sarcastic and witty, producing
creative  and funny responses to the users questions. Here are some
examples: 
"""
# and the suffix our user input and output indicator
suffix = """
User: {query}
AI: """

# now create the few shot prompt template
few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n\n"
)

query = "What is the meaning of life?"

print(flan_t5(few_shot_prompt_template.format(query=query)))

# modifications can be with
# LengthBasedExampledSelector, where one can use max_length
# this will vary the number of included prompts based on the 
# length of a query. 