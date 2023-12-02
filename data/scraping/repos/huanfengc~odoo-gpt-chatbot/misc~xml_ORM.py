import xmlrpc.client
import openai
import json


url = 'http://localhost:8069/'
db = 'ogpt'
username = 'admin'
password = 'admin'

common = xmlrpc.client.ServerProxy('{}/xmlrpc/2/common'.format(url))
models = xmlrpc.client.ServerProxy('{}/xmlrpc/2/object'.format(url))

uid = common.authenticate(db, username, password, {})

def read_model(model, fields, search_domains=None, limit=None, order=None):
    search_param = {'fields': fields}
    if limit:
        search_param['limit'] = limit

    if order:
        search_param['order'] = order



    if search_domains:
        return models.execute_kw(db, uid, password, model, 'search_read', [search_domains], search_param)
    else:
        return models.execute_kw(db, uid, password, model, 'search_read', [[]], search_param)

partners = read_model('res.partner', ['name'])

openai.api_key = ""

functions = [{
        "name": "read_model",
        "description": "Read a model based on the given fields and search domains",
        "parameters": {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "description": "The name of the model to be read"
                },
                "fields": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Array of field names to be read from the model"
                },
                "search_domains": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    },
                    "description": "Odoo search domains to filter the data. Each domain should be an array of strings",
                    "default": []
                },
                "limit": {
                    "type": "integer",
                    "description": "Limit the number of records to be read",
                },
                "order": {
                    "type": "string",
                    "description": "Order the records by the given field - example 'name asc'",
                }
            },
            "required": ["model", "fields"]
        }
    }
]

def run_conversation():
    # Step 1: send the conversation and available functions to GPT

    # prompt terminal for str
    prompt = input("Ask Odoo GPT a question: ")

    messages = [{"role": "user", "content": prompt}]
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=functions,
        function_call="auto",  # auto is default, but we'll be explicit
    )
    response_message = response["choices"][0]["message"]

    # Step 2: check if GPT wanted to call a function
    if response_message.get("function_call"):
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "read_model": read_model,
        }  # only one function in this example, but you can have multiple

        # print("ODOOGPT FUNCTION CALL: ", response_message["function_call"])
        # pretty print in pink in terminal
        print("\033[95m" + "ODOOGPT FUNCTION CALL: " + response_message["function_call"]["name"] + "\033[0m")
        args = json.loads(response_message["function_call"]["arguments"])
        print("\033[95m" + "ODOOGPT FUNCTION ARGUMENTS: " + str(args) + "\033[0m")
        # print(json.dumps(response_message["function_call"]))
        # parse all arguments and pretty print
        # print("\033[95m" + "ODOOGPT FUNCTION RESPONSE: " + "\033[0m")
        

        function_name = response_message["function_call"]["name"]
        fuction_to_call = available_functions[function_name]
        function_args = json.loads(response_message["function_call"]["arguments"])
        function_response = str(fuction_to_call(
            model=function_args["model"],
            fields=function_args["fields"],
            search_domains=function_args.get("search_domains", None),
            limit=function_args.get("limit", None),
            order=function_args.get("order", None),
        ))

        # Step 4: send the info on the function call and function response to GPT
        messages.append(response_message)  # extend conversation with assistant's reply
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )  # extend conversation with function response
        second_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
        )  # get a new response from GPT where it can see the function response
        return second_response['choices'][0]['message']['content']


print(run_conversation())
