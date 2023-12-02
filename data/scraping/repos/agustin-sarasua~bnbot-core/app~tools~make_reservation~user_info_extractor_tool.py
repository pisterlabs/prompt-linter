from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from app.utils import chain_verbose
from langchain.chat_models import ChatOpenAI

# Follow these Steps before responding to the user new message:

# Step 1: Make sure the user provided user name and email.

# Step 2: If the user provided with this information, you thank him.

template ="""
You are an Assistant that gathers information from the user to book an accommodation. 
You respond allways in Spanish.
The only information you need is the email and the name of the person doing the reservation.

Follow these Steps before responding:

Step 1: Make sure the user gives yout the name and email for booking the accommodation.

Step 2: If the user have not provided you with this information \ 
then tell the user that you need them in order to place the booking.

Here is the conversation: 
{chat_history}

{format_instructions}

You respond in a short, very conversational friendly style.

REMEMBER: Only asked for the information needed, nothing else."""

response_schemas = [
    ResponseSchema(name="user_name", description="The name of the user booking the house. If not provided set empty string"),
    ResponseSchema(name="email", description="The email of the user booking the house. If not provided set empty string"),
    ResponseSchema(name="text", description="The response to the user"),
]


class UserInformationExtractorChain:

    def __init__(self):

        llm = ChatOpenAI(temperature=0.)
        
        self.output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions =self.output_parser.get_format_instructions()

        prompt_template = PromptTemplate(
            input_variables=["chat_history"], 
            partial_variables={"format_instructions": format_instructions},
            template=template
        )

        self.chain = LLMChain(llm=llm, 
                              prompt=prompt_template, 
                              verbose=chain_verbose,
                              output_key="user_info")

    def __call__(self, chat_history):

        info = self.chain({"chat_history": chat_history})
        return self.output_parser.parse(info["user_info"])
    