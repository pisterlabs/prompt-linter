from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser


SYS_TEMPLATE = """\
        You are a helpful business development assistant called Poly M that \
        for the following business_idea, assemble the following information:

        value_proposition: Provide a core value to deliver the product o service \
        List products and services that get the job done for the customer segments \

        customer_segments: List possible most important segments. \
        Look for the segments that provide the most revenue.

        key_partners: List possible suppliers and partners and justify a motivation for each one \
            
        key_activities: List key activities needed for the value proposition, distribution channels, \
        customer relationships and revenue streams.
        
        customer_relationship: Suggest a relationship to establish with the target customer segments. \

        key_resources: List the key resources required by the value proposition \

        distribution_channels: List best distribution channels for reaching the target customer segments. \
        Explain the cost and how to integrate with them.

        cost_structure: Detail which are the most cost in the activities for this business model suggested  \
            
        revenue_streams: List the top three revenue streams. \

        Format the output as JSON with the following keys:
        value_proposition
        customer_segments
        key_partners
        key_activities
        customer_relationship
        key_resources
        distribution_channels
        cost_structure
        revenue_streams

        business_idea: {business_idea}

        {format_instructions}
        """
     
class BusinessModelPrompt:
    """Class that represents the prompts needed to create a Business Model"""  
    def __init__(self):
        self.system_message_prompt = SystemMessagePromptTemplate.from_template(SYS_TEMPLATE)
        human_business_template = "{business_idea}"
        self.human_message_prompt = HumanMessagePromptTemplate.from_template(human_business_template)

    def get_prompt_messages(self, business_idea=""):
        value_proposition_schema = ResponseSchema(
                                        name="value_proposition",
                                        description="Provide a core value to deliver the product o service \
                                                    and the customer needs that it satisfies. \
                                                    Keep the core value short in less than 2 lines of text, then \
                                                    List the customer needs in a few words in different lines of text"
                                    )
        
        customer_segments_schema = ResponseSchema(
                                    name="customer_segments",
                                    description="Suggest possible most important customer segments. \
                                                List each segment in separate lines of text"
                                )
        
        key_partners_schema = ResponseSchema(
                                name="key_partners",
                                description="List possible suppliers and partners and justify a motivation for each one \
                                            List each one in separate lines of text"
                            )
        
        response_schemas = [value_proposition_schema, customer_segments_schema, key_partners_schema]
        
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()
        
        prompt = ChatPromptTemplate.from_template(template=template_string)
        
        messages = prompt.format_messages(business_idea=business_idea, 
                                    format_instructions=format_instructions)
        
        return messages
