from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

# Define the prompt template for summarization
summary_template = '''Summarize the following text:\n{input_text}'''

summarization_prompt = PromptTemplate(
    input_variables=["input_text"],
    template=summary_template
)

# Format the summarization prompt
summarization_prompt.format(input_text="hello our world is 4th Dimension but some of the people don't think and talk Sh*** which make no sense")

# Initialize the language model and chain
llm = OpenAI(temperature=0.7)
summarization_chain = LLMChain(llm=llm, prompt=summarization_prompt)

# Run the summarization chain
summarization_chain.run(" Are Aliens real if they are then where are they on Earth, in space or in another universe where they are.")


#output
```
\n\nMany people believe that aliens exist, but the exact location of them is unknown. Some people theorize that they could be on Earth, in space, or even in another universe.```
