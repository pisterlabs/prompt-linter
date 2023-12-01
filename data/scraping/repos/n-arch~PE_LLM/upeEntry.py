from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate

# We will be using a chat model, defaults to gpt-3.5-turbo
from langchain.chat_models import ChatOpenAI

# To parse outputs and get structured data back
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

openai_api_key='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'
chat_model = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo', openai_api_key=openai_api_key)




def title_prompt(text):

    # The schema I want out
    response_schemas = [
        ResponseSchema(name="unit", description="The name of a pharmaceutical production unit"),
        ResponseSchema(name="productAbb", description="The abrevation of a pharmaceutical product"),
        ResponseSchema(name="object", description="The name of an object"),
        ResponseSchema(name="defect", description="The concise description of the defect")
    ]

    # The parser that will look for the LLM output in my schema and return it back to me
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    prompt = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template("Given a command from the user, extract the production unit and product abbriviation, the defective object and the defect \n \
                                                        {format_instructions}\n{user_prompt}")  
        ],
        input_variables=["user_prompt"],
        partial_variables={"format_instructions": format_instructions}
    )

    titel_query = prompt.format_prompt(user_prompt=text)
    prompt.format_prompt(user_prompt=user_prompt)
    titel_query = chat_model(titel_query.to_messages())
    output = output_parser.parse(titel_query.content)
    return output

#
#print (titel_query.messages[0].content)
#print (output)
#print (type(output))

#####################################Justification##############################################
# The schema I want out


def justification_prompt(justification_prompt):

    response_schemas_justifcation = [
        ResponseSchema(name="Entdeckung", description="How the defect was discoverd"),
        ResponseSchema(name="Situation", description="What is the Qualitiy Deveation that occured"),
        ResponseSchema(name="Zeit", description="The date and time the probelm occured"),
        ResponseSchema(name="Prozess", description="The process in which the defect occured"),
        ResponseSchema(name="Material", description="How much material was affected from the defect"),
        ResponseSchema(name="GMP", description="Why it is a Good Manufacturing Practices deviation"),
        ResponseSchema(name="AsIs", description="What the actual state is"),
        ResponseSchema(name="Target", description="What the target state is"),
        
    ]

    # The parser that will look for the LLM output in my schema and return it back to me
    output_parser_justification = StructuredOutputParser.from_response_schemas(response_schemas_justifcation)
    format_instructions_justification = output_parser_justification.get_format_instructions()


    prompt_justification = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template("Given a command from the user, \
            extract the information and rewrite it to be clear, concise, of high quality while retaining all the information and in german.\
            If the text doesn't contain the information write that the infoamtion is not available\
                    \n{format_instructions_justification}\n{user_prompt_justification}")  
        ],
        input_variables=["user_prompt_justification"],
        partial_variables={"format_instructions_justification": format_instructions_justification}
    )



    justification = prompt_justification.format_prompt(user_prompt_justification=justification_prompt)


    justification = chat_model(justification.to_messages())
    #output = output_parser1.parse(justification.content)
  
    justification_result_dict = output_parser_justification.parse(justification.content)
    outputstring = ""
    for key, value in justification_result_dict.items():
        outputstring += key + ": " + value + "\n"

    return outputstring


