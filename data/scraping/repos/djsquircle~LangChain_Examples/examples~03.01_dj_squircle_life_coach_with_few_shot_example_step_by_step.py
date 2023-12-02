# First things first, let's import the libraries we need
from dotenv import load_dotenv
from langchain import FewShotPromptTemplate, PromptTemplate, LLMChain
from langchain.llms import OpenAI

# This loads any environment variables in a .env file. In this case, we'll use it to load our OpenAI API key
load_dotenv()

# Here we're initializing a Language Learning Model (LLM) from OpenAI. We're using the text-davinci-003 model with a temperature of 0
# The temperature parameter controls the randomness of the model's output. A higher value like 0.9 makes the output more random, while a lower value like 0 makes it more deterministic
llm = OpenAI(model_name="text-davinci-003", temperature=0)

# These are examples of previous conversations that the model will use to understand the context of the conversation. They're all in surfer slang, in keeping with Dj Squircle's style
examples = [
    {
        "query": "What's the secret to happiness, Dj Squircle?",
        "answer": "Dude, happiness is like catching the perfect wave. It's all about waiting for the right moment and then riding it with all you've got!"
    }, {
        "query": "How can I become more productive, Dj Squircle?",
        "answer": "Productivity's like surfing, bro. You gotta balance your effort with chilling out. Can't ride waves non-stop, gotta rest and recharge too, you know?"
    }, {
        "query": "What's the meaning of life, Dj Squircle?",
        "answer": "Life, man, it's like the ocean. Sometimes it's chill, sometimes it's wild, but it's always beautiful. Just gotta learn to ride the waves and enjoy the ride!"
    }
]

# This is the template we'll use for our examples. It's a simple conversation format with the user asking a question and Dj Squircle giving an answer
example_template = """
User: {query}
Dj Squircle: {answer}
"""

# This is a PromptTemplate object. It's a way of telling the model how to format its inputs and outputs. In this case, it's using our example template and it expects a 'query' and an 'answer' for each example
example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

# This is a prefix that will be added to the start of our prompt. It gives some context to the conversation
prefix = """
The following are excerpts from conversations with Dj Squircle,
the AI with the vibes of a California surfer. He's always ready with a wave and a grin,
providing gnarly and often hilarious advice to the users' questions. Check it out, dude: 
"""

# This is a suffix that will be added to the end of our prompt. It tells the model what the user's query is
suffix = """
User: {query}
Dj Squircle: """

# This is a FewShotPromptTemplate object. It's a more complex type of prompt that uses a number of examples to help the model understand the context of the conversation
few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n\n"
)

# Here we're creating an LLMChain object. This is what we'll use to actually run our model
chain = LLMChain(llm=llm, prompt=few_shot_prompt_template)

# Now we're defining a new user query. This is the question that we're going to ask Dj Squircle
user_query = "Any tips for handling stress, Dj Squircle?"

# This is where we actually run the model. We're passing in our user query and the model will return Dj Squircle's response
response = chain.run({"query": user_query})

# Finally, we're printing out the user query and the model's response. This is what we'll see when we run the program
print("User Query:", user_query)
print("Dj Squircle:", response)
