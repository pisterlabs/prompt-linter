from langchain.chains import SequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import openai

# Set arguments
def PerformSequentialChainUsingChatGPT(dict_of_prompts,
                                       openai_api_key,
                                       temperture=.30,
                                       chat_model_name="gpt-3.5-turbo"):
    """This function takes a dictionary of prompts that are then sent to OpenAI's ChatGPT API in sequential order. The outputs of the earlier prompts can be used as inputs for later prompts. It returns the response from the sequential chain.
    
    Keyword arguments:
    - dict_of_prompts: A dictionary of prompts. The keys are the names of the outputs, and the values are the prompts. The first prompt should be the initial input.
    - openai_api_key: The API key for OpenAI's API.
    - temperture: The temperture, or "randomness", to use for the API call. Must be between 0 and 1. Defaults to .30.
    - chat_model_name: The name of the chat model to use. Defaults to "gpt-3.5-turbo".
    
    Return:
    - response: The response from the sequential chain.
    """
    
    # Get number of prompts in the prompt dictionary
    prompt_count = len(dict_of_prompts)

    # Create dictionary of prompt outputs
    dict_of_outputs = {}
    for i in range(prompt_count-1):
        dict_of_outputs["output_"+str(i)] = None

    # Create list of prompt chains, starting with initial input then iterating through the dictionary of outputs
    list_of_prompt_chains = []
    
    # Create the initial input prompt chain
    for i in range(prompt_count-1):
        # Get the prompt from the dictionary
        my_prompt_template = dict_of_prompts["output_"+str(i)]
        prompt_template = ChatPromptTemplate.from_template(my_prompt_template)
        
        # Set the model name and temperature
        chatgpt_model = ChatOpenAI(
            temperature=temperture, 
            model_name=chat_model_name,
            openai_api_key=openai_api_key
        )
                
        # Create the prompt chain
        prompt_chain = LLMChain(
            llm=chatgpt_model, 
            prompt=prompt_template,
            output_key="output_"+str(i),
        )
        
        # Add the prompt chain to the list of prompt chains
        list_of_prompt_chains.append(prompt_chain)

    # Create list of output variables
    list_of_output_variables = []
    for i in range(prompt_count-1):
        list_of_output_variables.append("output_"+str(i))

    # Create the sequential chain
    overall_chain = SequentialChain(
        chains=list_of_prompt_chains,
        input_variables=["initial_input"],
        output_variables=list_of_output_variables,
        verbose=True
    )

    # Run the sequential chain
    response = overall_chain(dict_of_prompts["initial_input"])
            
    # Return the response
    return(response)


# # Test the function
# PerformSequentialChainUsingChatGPT(dict_of_prompts={
#     "initial_input": """
#         Je trouve le goût médiocre. La mousse ne tient pas, c'est bizarre. 
#         J'achète les mêmes dans le commerce et le goût est bien meilleur...
#         Vieux lot ou contrefaçon !?
#     """,
#     "output_0": """
#         Translate the following review to English:
        
#         {initial_input}
#     """,
#     "output_1": """
#         Summarize the following review in 1 sentence:
        
#         {output_0}
#     """,
#     "output_2": """
#         What language is the following review:
        
#         {output_1}
#     """,
#     "output_3": """
#         Write a follow up response to the following summary in the specified language:
        
#         Summary: {output_0}
#         Language: {output_1}
#     """
#     },
#     openai_api_key = open("C:/Users/oneno/OneDrive/Desktop/OpenAI key.txt", "r").read()
# )

