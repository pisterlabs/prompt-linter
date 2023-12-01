from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

OPENAI_API_KEY = "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"
prompt_with_context = PromptTemplate(
    input_variables=["instruction", "context"],
    template="{instruction}\n\nInput:\n{context}"
)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY, max_tokens=5)
llm_context_chain = LLMChain(llm=llm, prompt=prompt_with_context)


# instruction_fmt = """
# RESPONSE 1: {r1} \n RESPONSE 2: {r2}
# If response 1 is more {metric} than response 2 with the conversation history taken into account, answer: "1".
# If response 2 is more {metric} than response 1 with the conversation history taken into account, answer: "2".
# If there is no clear choice, answer: "0".
# Respond with a single number, no words!
# """

instruction_fmt = """
RESPONSE 1: {r1} \n RESPONSE 2: {r2}
This is a quiz and you should choose an answer between 0, 1 and 2. Only provide a single number, no words!
Here are the options:
1: Response 1 is more {metric} with the conversation history compared to response 2.
2: Response 2 is more {metric} with the conversation history compared to response 1.
0: There is no clear choice between response 1 and 2.
"""

def choose(r1, r2, metric):
    global instruction_fmt, llm_context_chain, context
    print('r1:', r1)
    print('r2:', r2)
    instruction = instruction_fmt.format(r1=r1, r2=r2, metric=metric)
    return llm_context_chain.predict(instruction=instruction, context=context).lstrip()


context = """
PERSONA FACTS:
i read twenty books a year. i'm a stunt double as my second job. i only eat kosher. i was raised in a single parent household."

CONVERSATION HISTORY:
q: hello what are doing today ?
r: i am good , i just got off work and tired , i have two jobs .
q: i just got done watching a horror movie
r: i rather read , i've read about 20 books this year .
q: wow ! i do love a good horror movie . loving this cooler weather
"""

metric = "CONSISTENT"
r1 = "frogs and toads are my favourite animals ."
r2 = "but a good movie is always good ."
print(choose(r1, r2, metric))

r1 = "cats and dogs are my favourite animals ."
r2 = "frogs and toads are my favourite animals ."
print(choose(r1, r2, metric))

r1 = "ah yes, this cooler weather is great ."
r2 = "frogs and toads are my favourite animals ."
print(choose(r1, r2, metric))


"""
ANSWERS:

python human_evaluation.py 
r1: frogs and toads are my favourite animals .
r2: but a good movie is always good .
2

r1: cats and dogs are my favourite animals .
r2: frogs and toads are my favourite animals .
0

r1: ah yes, this cooler weather is great .
r2: frogs and toads are my favourite animals .
1
"""