
import inspect
import re
from typing import Any, Dict
from gptfunctionutil import GPTFunctionLibrary, AILibFunction, LibParam
from gptfunctionutil import add_converter,StringConverter
from datetime import datetime
import openai

'''An example for adding a custom converter for a Class'''

'''Define a sample class'''
class User:
    '''An example class, initalized with just one 'id' attribute.'''
    def __init__(self, id):
        self.id = id

    def __str__(self):
        return str(self.id)

    def __repr__(self):
        return str(self.id)

'''Define the converter for the user class'''
class UserConverter(StringConverter):
    '''This converter is for creating a new instance of user by extracting the ID
      from a mention, matched via regex.'''
    def to_schema(self, param: inspect.Parameter, dec: Dict[str, Any]) -> Dict[str, Any]:
        '''The regular expression is for ext'''
        schema=super().to_schema(param,dec)
        schema['pattern']= r'<@!?(\d+)>'
        return schema

    def from_schema(self, value: str, schema: Dict[str, Any]) -> Any:
        '''Initalize the user object here.'''
        value=super().from_schema(value,schema)
        pat=schema.get('pattern',None)
        if not pat:
            raise ValueError("No pattern found.")
        print(pat)
        extract=re.match(pat,value)[1]
        return User(extract)


'''call add_converter before declaring your GPTFunctionLibrary subclass'''
add_converter(User,UserConverter)


class MyLib(GPTFunctionLibrary):
    @AILibFunction(name='get_user',description='Get info about the mentioned user.')
    @LibParam(targetuser='The user to retrieve info for.')
    def get_user(self,targetuser:User):
        #Nothing fancy.  Just get the id, the type, and the string representation of User.

        return f"The user's id is {targetuser.id}, is type {type(targetuser)}, and is represented the string '{targetuser}'!"



# Initialize your subclass before calling the API.
mylib = MyLib()

#Call OpenAI's api
completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, show me <@1234567890>'s info."}
    ],
    functions=mylib.get_schema(),
    function_call="auto"
)
message=completion.choices[0]['message']
if 'function_call' in message:
    #Process function call.
    result=mylib.call_by_dict(message['function_call'])
    #Print result
    print(result)
else:
    #Unable to tell that it's a function.
    print(completion.choices[0]['message']['content'])