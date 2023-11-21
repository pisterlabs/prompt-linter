import os
from apikey import OPENAI_API_KEY
from langchain.llms import OpenAI
from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.chains import SimpleSequentialChain, SequentialChain
from langchain.chains import LLMChain

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# Text model example

llm = OpenAI(temperature=0.1)

# Example with SimpleSequentialChain
video_title_prompt_template = PromptTemplate(
    input_variables=["content"],
    template="""
        Write a title for a Youtube video about {content}
    """,
)

video_outline_prompt_template = PromptTemplate(
    input_variables=["title"],
    template="""
        Write a outline of a Youtube video about {title}. Output in the bullet list format.
    """,
)

overall_chain = SimpleSequentialChain(
    chains=[
        LLMChain(llm=llm, prompt=video_title_prompt_template),
        LLMChain(llm=llm, prompt=video_outline_prompt_template),
    ],
    verbose=True,
)

video_outline = overall_chain.run("Deep Learning in 1 minutes")
print(video_outline)

# Example with SequentialChain
video_title_prompt_template = PromptTemplate(
    input_variables=["content", "style"],
    template="""
        Write a title for a Youtube video about {content} with {style} style.
    """,
)

video_outline_prompt_template = PromptTemplate(
    input_variables=["title"],
    template="""
        Write a outline of a Youtube video about {title}. Output in the bullet list format. 
    """,
)

overall_chain = SequentialChain(
    chains=[
        LLMChain(llm=llm, prompt=video_title_prompt_template, output_key="title"),
        LLMChain(llm=llm, prompt=video_outline_prompt_template, output_key="outline"),
    ],
    input_variables=["content", "style"],
    output_variables=["title", "outline"],
    verbose=True,
)

video_outline = overall_chain(
    {"content": "Deep Learning in 1 minutes", "style": "funny"}
)
print(video_outline)
