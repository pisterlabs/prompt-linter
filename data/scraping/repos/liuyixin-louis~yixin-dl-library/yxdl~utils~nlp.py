import os
import openai
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import ChatPromptTemplate
import pandas as pd


_ = load_dotenv(find_dotenv('/data/yixin/workspace/yixin-dl-library/yxdl/.env')) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']



def open_ai_get_completion_simple(prompt, model="gpt-3.5-turbo", system_prompt = "You are a helpful chatbot that helps people to address their problem", temperature=0, api_key=None):
    openai.api_key = api_key if api_key else os.environ['OPENAI_API_KEY']
    messages = [
        {
            "role": "system", "content": system_prompt,
        },
        
        {
        "role": "user", "content": prompt
        },
        ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message["content"]


def answer_with_prompt_template(prompt_template, chat_model = ChatOpenAI(temperature=0.0), **kwargs):
    inp_dict = kwargs
    inp_slot = prompt_template.messages[0].prompt.input_variables
    call_dict = {slot: '' for slot in inp_slot}
    for slot in inp_slot:
        if slot in inp_dict:
            call_dict[slot] = inp_dict[slot]
    message_from_template = prompt_template.format_messages(**call_dict)
    return chat_model(message_from_template)

def get_respone():
    
    template_string = """Translate the text \
        that is delimited by triple backticks \
        into a style that is {style}. \
        text: ```{text}```
    """
    prompt_template = ChatPromptTemplate.from_template(template_string)
    style = "chinese"
    text = "today is a good day"
    reponse = answer_with_prompt_template(prompt_template, style=style, text=text)    
    print(reponse.content)


def parse_structure_output(chat=ChatOpenAI(temperature=0.0)):
    from langchain.output_parsers import ResponseSchema
    from langchain.output_parsers import StructuredOutputParser
    emotion_schema = ResponseSchema(name="emotion",
                                    description="What is the emotion of the text?\
                                        it can be one of the following: \
                                            natural, positive, negative, \
                                        ")
    gift_schema = ResponseSchema(name="gift",
                             description="Was the item purchased\
                             as a gift for someone else? \
                             Answer True if yes,\
                             False if not or unknown.")
    delivery_days_schema = ResponseSchema(name="delivery_days",
                                        description="How many days\
                                        did it take for the product\
                                        to arrive? If this \
                                        information is not found,\
                                        output -1.")
    price_value_schema = ResponseSchema(name="price_value",
                                        description="Extract any\
                                        sentences about the value or \
                                        price, and output them as a \
                                        comma separated Python list.")

    response_schemas = [emotion_schema,
                        gift_schema, 
                        delivery_days_schema,
                        price_value_schema]
    
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    
    format_instructions = output_parser.get_format_instructions()

    customer_review = """\
    This leaf blower is pretty amazing.  It has four settings:\
    candle blower, gentle breeze, windy city, and tornado. \
    It arrived in two days, just in time for my wife's \
    anniversary present. \
    I think my wife liked it so much she was speechless. \
    So far I've been the only one using it, and I've been \
    using it every other morning to clear the leaves on our lawn. \
    It's slightly more expensive than the other leaf blowers \
    out there, but I think it's worth it for the extra features.
    """


    review_template_2 = """\
    For the following text, extract the following information:

    emotion : What is the emotion of the text? \
    it can be one of the following: natural, positive, negative, 
    
    gift: Was the item purchased as a gift for someone else? \
    Answer True if yes, False if not or unknown.

    delivery_days: How many days did it take for the product\
    to arrive? If this information is not found, output -1.

    price_value: Extract any sentences about the value or price,\
    and output them as a comma separated Python list.

    text: {text}

    {format_instructions}
    """

    prompt = ChatPromptTemplate.from_template(template=review_template_2)

    messages = prompt.format_messages(text=customer_review, 
                                    format_instructions=format_instructions)
        
    response = chat(messages)
    
    # structured_output = output_parser(response)
    output_dict = output_parser.parse(response.content)
    print(output_dict)
    
def simple_llm_chain():
    df = pd.read_csv('/data/yixin/workspace/yixin-dl-library/yxdl/courses/langchain-deepai/Chain/Data.csv')
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    from langchain.chains import LLMChain
    
    llm = ChatOpenAI(temperature=0.9)
    prompt = ChatPromptTemplate.from_template(
        "What is the best name to describe \
        a company that makes {product}?"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    product = "Queen Size Sheet Set"
    chain.run(product)
    
    second_prompt = ChatPromptTemplate.from_template(
        "Write me a short description of the company that makes {product}."
    )
    
    chain_two = LLMChain(llm=llm, prompt=second_prompt)
    
    from langchain.chains import SimpleSequentialChain
    overall_simple_chain = SimpleSequentialChain(
        chains=[chain, chain_two], verbose=True)
    
    product="thin condom"
    overall_simple_chain.run(product)
    
def mulitple_llm_chains_with_variable_dependences():
    import warnings
    warnings.filterwarnings('ignore')
    from langchain.chains import LLMChain, SequentialChain
    llm = ChatOpenAI(temperature=0.9)
    first_pormpt = ChatPromptTemplate.from_template(
        "What is the best name to describe \
            {Product}?"
    )
    chain_one = LLMChain(llm=llm, prompt=first_pormpt, output_key="Product_name")
    second_prompt = ChatPromptTemplate.from_template(
        "Write me a short description of the company that makes {Product_name}.")
    chain_two = LLMChain(llm=llm, prompt=second_prompt, output_key="Company_description")
    third_prompt = ChatPromptTemplate.from_template(
        "Base on the given company name and description, \
            please provide some selling strategies, \n\
                product name: {Product_name},\n\
                    product description: {Company_description}")
    chain_three = LLMChain(llm=llm, prompt=third_prompt, output_key="Selling_strategies")   
    fourth_prompt = ChatPromptTemplate.from_template(
        """
        Give me some idea on how to make good ads for the product {Product_name}.
        The following is some selling strategies that our company has been using:
        {Selling_strategies}. 
        """
    )
    chain_four = LLMChain(llm=llm, prompt=fourth_prompt, output_key="Ads")
    overall_chain = SequentialChain(chains=[chain_one, chain_two, chain_three, chain_four], input_variables=['Product', ],                                                 output_variables=['Product_name', 'Company_description', 'Selling_strategies', 'Ads'], verbose=True)
    print(
        overall_chain("thin condom")
    )
    
    
def multiple_chains_with_router():
    physics_template = """You are a very smart physics professor. \
    You are great at answering questions about physics in a concise\
    and easy to understand manner. \
    When you don't know the answer to a question you admit\
    that you don't know.

    Here is a question:
    {input}"""


    math_template = """You are a very good mathematician. \
    You are great at answering math questions. \
    You are so good because you are able to break down \
    hard problems into their component parts, 
    answer the component parts, and then put them together\
    to answer the broader question.

    Here is a question:
    {input}"""

    history_template = """You are a very good historian. \
    You have an excellent knowledge of and understanding of people,\
    events and contexts from a range of historical periods. \
    You have the ability to think, reflect, debate, discuss and \
    evaluate the past. You have a respect for historical evidence\
    and the ability to make use of it to support your explanations \
    and judgements.

    Here is a question:
    {input}"""


    computerscience_template = """ You are a successful computer scientist.\
    You have a passion for creativity, collaboration,\
    forward-thinking, confidence, strong problem-solving capabilities,\
    understanding of theories and algorithms, and excellent communication \
    skills. You are great at answering coding questions. \
    You are so good because you know how to solve a problem by \
    describing the solution in imperative steps \
    that a machine can easily interpret and you know how to \
    choose a solution that has a good balance between \
    time complexity and space complexity. 

    Here is a question:
    {input}"""
    
    prompt_infos = [
    {
        "name": "physics", 
        "description": "Good for answering questions about physics", 
        "prompt_template": physics_template
    },
    {
        "name": "math", 
        "description": "Good for answering math questions", 
        "prompt_template": math_template
    },
    {
        "name": "History", 
        "description": "Good for answering history questions", 
        "prompt_template": history_template
    },
    {
        "name": "computer science", 
        "description": "Good for answering computer science questions", 
        "prompt_template": computerscience_template
    }
    ]
    from langchain.chains import LLMChain
    from langchain.chains.router import MultiPromptChain
    from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser
    from langchain.prompts import PromptTemplate
    
    llm = ChatOpenAI(temperature=0)
        
    destination_chains = {}
    for p_info in prompt_infos:
        name = p_info["name"]
        prompt_template = p_info["prompt_template"]
        prompt = ChatPromptTemplate.from_template(template=prompt_template)
        chain = LLMChain(llm=llm, prompt=prompt)
        destination_chains[name] = chain  
        
    destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
    destinations_str = "\n".join(destinations)
    
    default_prompt = ChatPromptTemplate.from_template("{input}")
    default_chain = LLMChain(llm=llm, prompt=default_prompt)
    MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a \
    language model select the model prompt best suited for the input. \
    You will be given the names of the available prompts and a \
    description of what the prompt is best suited for. \
    You may also revise the original input if you think that revising\
    it will ultimately lead to a better response from the language model.

    << FORMATTING >>
    Return a markdown code snippet with a JSON object formatted to look like:
    ```json
    {{{{
        "destination": string \ name of the prompt to use or "DEFAULT"
        "next_inputs": string \ a potentially modified version of the original input
    }}}}
    ```

    REMEMBER: "destination" MUST be one of the candidate prompt \
    names specified below OR it can be "DEFAULT" if the input is not\
    well suited for any of the candidate prompts.
    REMEMBER: "next_inputs" can just be the original input \
    if you don't think any modifications are needed.

    << CANDIDATE PROMPTS >>
    {destinations}

    << INPUT >>
    {{input}}

    << OUTPUT (remember to include the ```json)>>"""
    router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
        destinations=destinations_str
    )
    router_prompt = PromptTemplate(
        template=router_template,
        input_variables=["input"],
        output_parser=RouterOutputParser(),
    )

    router_chain = LLMRouterChain.from_llm(llm, router_prompt)
        
    chain = MultiPromptChain(router_chain=router_chain, 
                         destination_chains=destination_chains, 
                         default_chain=default_chain, verbose=True
                        )
    print(
        chain("What is the speed of light?")
    )
    return chain

    
if __name__ == "__main__":
    # get_respone()
    # parse_structure_output()
    # simple_llm_chain()
    # mulitple_llm_chains_with_variable_dependences()
    multiple_chains_with_router()
    
    