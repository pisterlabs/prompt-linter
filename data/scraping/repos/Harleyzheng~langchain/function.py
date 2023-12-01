from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

functions = [
  {
    "name": "solver",
    "description": "Formulates and solves an equation",
    "parameters": {
      "type": "object",
      "properties": {
        "equation": {
          "type": "string",
          "description": "The algebraic expression of the equation"
        },
        "solution": {
          "type": "string",
          "description": "The solution to the equation"
        }
      },
      "required": ["equation", "solution"]
    }
  }
]

# Need gpt-4 to solve this one correctly
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Write out the following equation using algebraic symbols then solve it."),
        ("human", "{equation_statement}")
    ]
)
model = ChatOpenAI(model="gpt-4", temperature=0).bind(function_call={"name": "solver"}, functions=functions)
runnable = (
    {"equation_statement": RunnablePassthrough()} 
    | prompt 
    | model
)
print(runnable.invoke("x raised to the third plus seven equals 12"))