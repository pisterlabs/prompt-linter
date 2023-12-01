from langchain.llms import OpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import OpenAIModerationChain

# def check_moderation(query,api_key):
#     moderation_chain = OpenAIModerationChain(openai_api_key=api_key,error=True)
#     moderation_chain.run(query)

# class CustomModeration(OpenAIModerationChain):
    
    
#     def _moderate(self, text: str, results: dict) -> str:
#         categories = results.get("categories", {})
#         flagged = results.get("flagged", False)
        
        
#         # Check for specific moderation conditions using OR conditions
#         if (
#             flagged or
#             categories.get("harassment/threatening", False) or
#             (categories.get("violence", False) or categories.get("violence/graphic", False)) or
#             (categories.get("self-harm/intent", False) or categories.get("self-harm/instructions", False))
#         ):
#             error_str = "Content violates moderation policies."
#             return error_str
        
#         # Add more conditions as needed
        
#         return text



class CustomModeration(OpenAIModerationChain):
    
    def __init__(self, openai_api_key):
        super().__init__(openai_api_key=openai_api_key)
    
    
    def _moderate(self, text: str, results: dict) -> str:
        # Check if the query meets certain conditions
        # You can define your custom conditions here
        if "specific_condition" in text:
            error_str = "The query violates a specific moderation policies."
            return error_str
        
        # Check if the query is flagged or any other category is set to true
        flagged = results.get("flagged", False)
        categories = results.get("categories", {})
        
        if flagged or any(categories.values()):
            error_str = "Content violates one or more moderation policies."
            return error_str
        
        # If no error conditions are met, return None
        return None




def prompt_analysis(query, api_key, temp, max_token):  # prompt_enity?
    llm = OpenAI(temperature=temp, max_tokens=max_token, openai_api_key=api_key)
    prompt_template = """
    Analyze the provided prompt {query} and assign a score from 1 to 100 using the following guidelines:
        - Ensure the given prompt is clear, specific, and presented in complete sentences.
        - Include relevant context within the prompt; insufficient context may result in a lower score.
        - Accommodate any system or user instructions that may be present in the prompt.
        - Take into consideration explicit cues that might be included.
        - If the task is complex, break down the prompt or set limitations to streamline the scope.
        Based on these criteria, craft a new_prompt that adheres to all the specified requirements, {format_instructions}
    """

    response_schemas = [
        ResponseSchema(
            name="score", description="this is the score for the given prompt"
        ),
        ResponseSchema(
            name="new_prompt", description="this is the new prompt for the given prompt"
        ),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    format_instructions = output_parser.get_format_instructions()

    example_prompt = PromptTemplate(
        input_variables=["query"],
        template=prompt_template,
        partial_variables={"format_instructions": format_instructions},
    )

    _input = example_prompt.format_prompt(query=query)
    output = llm(_input.to_string())

    x = output_parser.parse(output)

    score = x["score"]
    new_prompt = x["new_prompt"]

    return score, new_prompt
