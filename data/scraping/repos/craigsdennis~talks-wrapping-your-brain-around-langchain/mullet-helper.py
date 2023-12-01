from langchain.chains import create_tagging_chain
from langchain.chat_models import ChatOpenAI

chain = create_tagging_chain(llm=ChatOpenAI(temperature=0), schema={
    "properties": {
        "area": {
            "type": "string",
            "description": "Understand which area the user is talking about"
        },
        "item": {
            "type": "string",
            "description": "Determine which item the user is attempting to obtain"
        },
        "frustration_level": {
            "type": "integer",
            "enum": [1, 2, 3, 4, 5],
            "description": "The lower the number the less frustrated they are."
        }
    }
}, verbose=True)

response = chain.run("I cannot find the fricking Purah Pad. I am in the middle of Hyrule field. Where is it?!!!!")
print(response)