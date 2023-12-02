from typing import List

from pydantic import BaseModel, Field

from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chains.openai_functions import create_structured_output_chain
from langchain import LLMChain

from data_integration_questionnaire.config import cfg
from data_integration_questionnaire.log_init import logger

QUESTIONNAIRE_SYSTEM_MESSAGE = "You are an expert data integration and gouvernance expert that can give advice based on a questionnaire"
QUESTIONNAIRE_START = "=== 'QUESTIONNAIRE START' ==="
QUESTIONNAIRE_END = "=== 'QUESTIONNAIRE END' ==="

BEST_PRACTICES = """
1. No code/ Low Code
Capability: There are more and more tools that are emerging in market making it easier to do data integration between systems without writing any code and use out of box connectors. 
The vast array of connectors gives organization agility to integrate with systems. 
Some of the most popular connectors include snowflake and salesforce connectors besides database connectors. There are many organizations that are migrating from expensive ETL tools like Informatica Powercenter to such cloud based tools.
Enabler: Such ease in integration gives the capability makes it easy to consume data downstream for applications and apply reverse ETL.

Low code and no code tools can speed up the integration of multiple systems.

2. Integration with Data Catalog
Capability: Are data moves across multiple systems, there is an increasing need to capture the metadata such as data lineage and exported to data catalog. 
There is also increasing demand to visualize data lineage and discover underlying relationships to improve data literacy.

Enabler: Using both data catalogs and data lineage, organizations can ensure data accuracy, implement data governance, and manage business change.

Data Catalogs can help organisations to document and understand better their data, thus helping firms and companies to improved the consistency and accuracy of their data.

3. CDC (Change Data Capture)
Capability: Publish changes in data as and when changes occur. Think event instead of schedule to minimize data latency.

Enabler: Minimizing data latency often leads to providing more timely transparency/visibility to business processes and thereby taking timely business decisions.

Change data capture is ideal when you want the data to be kept up to date all the time. It is a powerful tool to keep data consistency and keep all data related systems synchronized.

4. Open Source Abstraction on Compute Engine

Since organization have compute engines such as snowflake and databaricks already in their tech stack, there is a need to either use those compute engines for ETL operations or use an abstraction later top of these compute engines. 
Tools such as dbt (data build tool) is an open-source command-line tool that helps data analysts and engineers build data models and implement data pipelines. 
Another open-source tool Apache SeaTunnel is among the top 10 Apache projects for data integration. Tools such as Pentaho and Talend also can be leveraged.

ETL tools are ideal for building data pipelines and convert data into more suitable formats for data analysis and reporting.

5. Unified Integration at scale

Capability: There is an emerging trend where there is a need to do integration with a single tool whether its bulk/batch data or real time or streaming data. 
Organizations have multiple tools for data integration where ETL us used for bulk loads, Kafka used for real time publish, Nifi for streamlining. Some of the tools are on-prem and there is trend to move these workloads and data integration capabilities on cloud. 

Enabler: Having a unified platform simplifies maintainability of the integration platform.

Unified Data Integration- Why the Whole is Greater Than the Sum
Top companies now recognize the need for a more unified integration approach that combines the right technologies and managed services to deliver a more consistent and reliable view of their data across disparate applications, ultimately driving measurable business results.
Today’s enterprise data environments can be a goldmine of insight or a quagmire of confusion depending on the company and their approach to data integration and data management. 
Many struggle to manage a complex web of cloud applications — ironically designed to alleviate the very problems they now face with data accessibility and timeliness of delivery.
"""

HUMAN_MESSAGE = f"""Please give a customer some advice to improve the data integration based on the answers to the questionnaire. 
The questionnnaire starts with {QUESTIONNAIRE_START} and ends with {QUESTIONNAIRE_END}.

{QUESTIONNAIRE_START}
{{questionnaire}}
{QUESTIONNAIRE_END}

Here are some best practices that you can mention in your answer depending on the results of the questionnaire:

{BEST_PRACTICES}


"""

CLASSIFICATION_QUESTIONNAIRE_SYSTEM_MESSAGE = "You are an expert data integration and gouvernance expert that can classify customers according to their quizz answers"
CLASSIFICATION_TEMPLATE_HUMAN_MESSAGE = f"""Please classify a customer according to the answers of the questionnaire. 
The questionnnaire starts with {QUESTIONNAIRE_START} and ends with {QUESTIONNAIRE_END}.

{QUESTIONNAIRE_START}
{{questionnaire}}
{QUESTIONNAIRE_END}

If the replies of the customer indicate a great level of confidence, you can classify this customer as "Advanced".

For example if the customer answers this question 'Does your organization support an event driven architecture for data integration?' positively,
the customer cannot be a 'Beginner'

If the customer answers this question here positively: 'Does your organization export data lineage to data catalog?',
the customer has at least a 'Professional' level.

"""


class AdviceProfile(BaseModel):
    """Contains the information on how a candidate matched the profile."""

    advices: List[str] = Field(
        ...,
        description="The list of advices given based on best practices.",
    )

class DataIntegrationClassificationProfile(BaseModel):
    """Gives a classification to a customer based on his answers"""

    classification: List[str] = Field(
        ...,
        description="The classification of the customer on his journey base on the answers to a quizz",
        enum=["Beginner", "Intermediate", "Professional", "Advanced", "Expert"]
    )



def prompt_factory() -> ChatPromptTemplate:
    prompt_msgs = [
        SystemMessage(content=QUESTIONNAIRE_SYSTEM_MESSAGE),
        HumanMessagePromptTemplate.from_template(HUMAN_MESSAGE),
    ]
    return ChatPromptTemplate(messages=prompt_msgs)


def prompt_factory_classification() -> ChatPromptTemplate:
    prompt_msgs = [
        SystemMessage(content=CLASSIFICATION_QUESTIONNAIRE_SYSTEM_MESSAGE),
        HumanMessagePromptTemplate.from_template(CLASSIFICATION_TEMPLATE_HUMAN_MESSAGE),
    ]
    return ChatPromptTemplate(messages=prompt_msgs)


def create_match_profile_chain_pydantic() -> LLMChain:
    return create_structured_output_chain(
        AdviceProfile,
        cfg.llm,
        prompt_factory(),
        verbose=cfg.verbose_llm,
    )


def create_classification_profile_chain_pydantic() -> LLMChain:
    return create_structured_output_chain(
        DataIntegrationClassificationProfile,
        cfg.llm,
        prompt_factory_classification(),
        verbose=cfg.verbose_llm,
    )


def create_input_dict(questionnaire: str) -> dict:
    return {"questionnaire": questionnaire}


def extract_advices(chain_res: AdviceProfile) -> List[str]:
    return chain_res.advices


