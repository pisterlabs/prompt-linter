from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.schema.runnable import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
prompt = ChatPromptTemplate.from_template("please tel me a joke about a {role}")

functions = [{
    "name": "joke",
    "description": "the model tells you a joke about a particular person",
    "parameters": {
        "type": "object",
        "properties": {
            "setup": {
                "type": "string",
                "description": "the setup of the joke"},
            "punchline": {
                "type": "string",
                "description": "the punchline of the joke"},
            "explanation": {
                "type": "string",
                "description": "why is the joke funny?" 
                }                                                        
            },
            "required": ["setup", "punchline", "explanation"]
        },
    }]
# the llm will consider the function and prepare it answers to fill up each properties. The description is actually a prompt.
# the returned content is actually empty. There is no content, the answee is a json in the additional_kwargs field. (if we use ChatOpenAI.predict_messages)
#  runnable pass through will pick up the variable when the chain is invoked

chain = {"role": RunnablePassthrough()} | prompt | model.bind(function_call={"name": "joke"}, functions=functions) | JsonOutputFunctionsParser(key_name="setup")
response = chain.invoke("sarcastic doctor")
print(response)
