def setup_environment():
    """
    Load environment variables.
    """
    from dotenv import load_dotenv
    load_dotenv()

def initialize_llm():
    """
    Initialize Language Learning Model (LLM) with model name and temperature.
    """
    from langchain.llms import OpenAI
    return OpenAI(model_name="text-davinci-003", temperature=0)

def setup_few_shot_prompt_template(examples, example_template):
    """
    Set up Few Shot Prompt Template with examples, example prompt, prefix, and suffix.
    """
    from langchain import FewShotPromptTemplate, PromptTemplate
    
    example_prompt = PromptTemplate(
        input_variables=["query", "answer"],
        template=example_template
    )

    prefix = """
    The following are excerpts from conversations with Dj Squircle,
    the enigmatic AI turned life coach. Always ready with a beat and a grin,
    he provides insightful and often hilarious advice to the users' questions.
    Here are some examples: 
    """

    suffix = """
    User: {query}
    Dj Squircle: """

    return FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["query"],
        example_separator="\n\n"
    )

def run_query(user_query, chain):
    """
    Run the LLMChain for the user query and print the results.
    """
    response = chain.run({"query": user_query})

    print("User Query:", user_query)
    print("Dj Squircle:", response)

# Setup environment
setup_environment()

# Initialize LLM
llm = initialize_llm()

# Define examples and example template
examples = [
    {
        "query": "What's the secret to happiness, Dj Squircle?",
        "answer": "Well, mate, happiness is like a sick beat. It's all about finding your rhythm and dancing to it, no matter what."
    }, {
        "query": "How can I become more productive, Dj Squircle?",
        "answer": "Productivity, huh? Try to think of it like a playlist. Some tracks are fast, some are slow. Find the right mix for you, and let it play!"
    }, {
        "query": "What's the meaning of life, Dj Squircle?",
        "answer": "Life's like a song, mate. Sometimes it's fast, sometimes it's slow, but it's always moving. So keep dancing, keep laughing, and make it a banger!"
    }
]

example_template = """
User: {query}
Dj Squircle: {answer}
"""

# Setup few-shot prompt template
few_shot_prompt_template = setup_few_shot_prompt_template(examples, example_template)

# Create the LLMChain for the few-shot prompt template
from langchain import LLMChain
chain = LLMChain(llm=llm, prompt=few_shot_prompt_template)

# Define the user query
user_query = "Any tips for handling stress, Dj Squircle?"

# Run the query
run_query(user_query, chain)
