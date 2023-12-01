from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, ChatMessage #this is like a moderator message
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

function_descriptions =[
    {
        "name": "edit_financial_forecast",
        "description": "make an edit to the financial model for the user",
        "parameters": {
            "type": "object",
            "properties": {
                "year": {
                    "type": "string",
                    "description": "the year of the forecast"},
                "category": {
                    "type": "string",
                    "description": "the category that the user wnats to edit"},
                "amount": {
                    "type": "integer",
                    "description": "the amount that the user wants to edit"
                }
            },
            "required": ["year", "category", "amount"]
        }
    },
    {
        "name": "print_financial_forecast",
        "description": "send the financial forecast to the printer",
        "parameters": {
            "type": "object",
            "properties": {
                "printer_name": {
                    "type": "string",
                    "description": "the name of the printer that the forecast shoul be ",
                    "enum": ["home_printer", "office_printer"]
                }
            },
            "required": ["printer_name"]
        }
    }
]


user_request = """please do 3 things: add 40 units to 2023 headcount and substract 23 units from 2022 opex, 
then print out forecast at my home"""

#the AI will pick the right function, in this case it will be the edit_financial_forecast function
first_response = model.predict_messages([HumanMessage(content=user_request)], functions=function_descriptions)

#print(first_response.additional_kwargs)
# the content field of the returned message is empty.
# content=''
# addiontal_kwargs= {'function_call': {'name': 'edit_financial_forecast', 
#                   'arguments': '{\n  
#                                   "year": "2023",\n  
#                                   "category": "headcount",\n  
#                                   "amount": 40\n}'}

second_response = model.predict_messages([HumanMessage(content=user_request),
                                          AIMessage(content=first_response.additional_kwargs["function_call"]["arguments"]),
                                          ], 
                                         functions=function_descriptions,
                                         function_call="auto")

# ChatMessage(role="function", additional_kwargs={"name": "edit_financial_forecast"},
#                                                       content="Just updated the financial forecast for 2023 headcount")

third_response = model.predict_messages([HumanMessage(content=user_request),
                                          AIMessage(content=first_response.additional_kwargs["function_call"]["arguments"]),
                                          AIMessage(content=second_response.additional_kwargs["function_call"]["arguments"]),
                                          ], 
                                         functions=function_descriptions,
                                         function_call="auto")
final_response = model.predict_messages([HumanMessage(content=user_request),
                                          AIMessage(content=first_response.additional_kwargs["function_call"]["arguments"]),
                                          AIMessage(content=second_response.additional_kwargs["function_call"]["arguments"]),
                                          AIMessage(content=third_response.additional_kwargs["function_call"]["arguments"]),
                                          ChatMessage(role='function',
                                                    additional_kwargs = {'name': "print_financial_forecast"},
                                                    content = """
                                                        just printed the document at home
                                                    """)
                                        ], 
                                         functions=function_descriptions,
                                         function_call="auto")

print(final_response)