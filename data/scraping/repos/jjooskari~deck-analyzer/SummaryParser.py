import os
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from pydantic import BaseModel, Field
from langchain.callbacks import get_openai_callback
from datetime import date

# ! Could try to not use parser, but instead use the json directly provided by GPT

class Summary(BaseModel):
    companyName: str = Field(description="Name of the company.")
    stage: str = Field(description="Current funding stage of the company, or N/A")
    contact: str = Field(description="Primary contact at the company, e.g. CEO, or N/A")
    website: str = Field(description="Official website of the company or N/A")
    date: str = Field(default_factory=lambda: date.today().strftime("%B %d, %Y"), description="Current date")
    summary: str = Field(description="Brief summary of the company's business, product & deal")
    problem: str = Field(description="Problem the company is addressing.")
    productBusinessModel: str = Field(description="Description of the product and the business model.")
    usp: str = Field(description="Unique selling proposition of the company's product.")
    marketCompetition: str = Field(description="Information about the market (size) and competition if available.")
    customersSales: str = Field(description="Details about customers and sales traction.")
    productRoadmap: str = Field(description="Product maturity and future roadmap or N/A")
    teamManagement: str = Field(description="Information about the team and management or N/A")
    impact: str = Field(description="Assessment of the company's impact or N/A")
    investors: str = Field(description="Existing investors and the company's board or N/A")
    financials: str = Field(description="Company's revenue, projections, or N/A")
    technologyIP: str = Field(description="Possible details on technology and intellectual properties or N/A")
    regulatoryRisksOps: str = Field(description="Possible regulatory risks and opportunities or N/A")
    dealStructureTerms: str = Field(description="Terms and details about the deal structure and timeline, or N/A")

def parse_summary(summary):
    parser = PydanticOutputParser(pydantic_object=Summary)

    format_instructions = parser.get_format_instructions()

    prompt = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template(
                """Your task is change the format of the pitchdeck summary.
    {format_instructions}
    Unformatted pitchdeck summary: {summary}"""
            )
        ],
        input_variables=['summary'],
        partial_variables={
            'format_instructions': format_instructions
        }
    )

    chat_model = ChatOpenAI(
        model="gpt-3.5-turbo",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        verbose=True
    )

    _input = prompt.format_prompt(summary=summary)

    with get_openai_callback() as cost:
        output = chat_model(_input.to_messages())
        parsed = parser.parse(output.content)
        print(f"\n\nFinished parsing, additional cost: \n\n{cost}\n\nParsing output:\n")
        print(parsed.dict())

    return parsed.dict()