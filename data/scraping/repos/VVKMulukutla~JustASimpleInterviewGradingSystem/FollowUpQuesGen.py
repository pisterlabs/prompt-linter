from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

def question_generation(question, ans, context):
    # Define the prompt template for generating follow-up questions
    PROMPT_TEMPLATE = """Answer the following question:
    {question}
    Here is the user given answer:
    {answer}
    Based on the following context:
    {context}
    -
    what are the top 3 follow-up questions to ask with the intent to validate the genuinity of the answer and to gather more insights about the candidate in relation to the context?
    """
    
    # Create a PromptTemplate object with the input variables and template
    PROMPT = PromptTemplate(input_variables=["question", "answer", "context"], template=PROMPT_TEMPLATE)
    
    # Create an OpenAI language model (LLM) instance
    llm = OpenAI(model_name="text-davinci-003",
                 temperature=0.7,
                 max_tokens=100,
                 top_p=1.0,
                 frequency_penalty=0.0,
                 presence_penalty=0.0)
    
    # Create an LLMChain instance with the LLM and prompt template
    chain = LLMChain(llm=llm, prompt=PROMPT)
    
    # Apply the chain 
    chain.apply
    
    # Predict and parse the follow-up questions using the chain
    q = chain.predict_and_parse(question=question, answer=ans, context=context)
    
    # Return the generated follow-up questions
    return q
