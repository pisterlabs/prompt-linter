import main
import json
import re
from langchain.prompts import PromptTemplate
from langchain.llms import VertexAI
from langchain.chains import LLMChain


def Parser(sorted_files):
    """
    Extracts various entity details from the sorted pages of a minute book.

    Args:
        sorted_files (list): A list of tuples where each tuple contains a page number and file name.

    Returns:
        A list of dictionaries where each dictionary contains the extracted entity details that match the minute book
        extraction schema. Each detail object includes the date, extracted details, and the URL of the source document
        page where the details were extracted from.
    """

    entity_name = ""
    entity_details = []

    for file in sorted_files:
        page_number, file_name = file
        content = main.get_page(file_name)
        lowercase_content = content.lower()

        #  "entity_name": string, // Incorporation number for the corporation
        if page_number == 1:
            entity_name = extract_entity_name(content)

        #  "tax_id_number": string, // Tax identification number for the corporation
        if "business number" in lowercase_content or "business no." in lowercase_content:
            tax_id_number = extract_tax_id_number(content)
            if tax_id_number is not None:
                entity_details.append({"tax_id_number": tax_id_number, "provenance": main.get_url(file_name)})

        #  "entity_number": string, // Incorporation number for the corporation
        #  "entity_type": string // Type of business entity
        #  "formation_date": string, // Date (YYYY-MM-DD) when the corporation was incorporated
        #  "address": string, // Address where the corporation is registered
        #  "home_jurisdiction": string, // Jurisdiction where the corporation is incorporated
        if "certificate" not in lowercase_content and "articles" in lowercase_content and ("address" in lowercase_content or "number" in lowercase_content):
            try:
                output = extract_entity_details(content)
                output = json.loads(output)
                output['entity_name'] = output['entity_name'].upper()

                missing_values = False
                for key, value in output.items():
                    if not value:
                        missing_values = True
                        break

                if output['entity_name'] == entity_name and not missing_values and "address" in output:
                    entity_details.append({"details": output, "provenance": main.get_url(file_name)})

            except json.decoder.JSONDecodeError:
                pass

        # TODO implement Fiscal Month, Fiscal Day, Home Report Filed Date, and Waived Auditor?

    return entity_details


def extract_entity_name(content):
    prompt = PromptTemplate(
        input_variables=["content"],
        template="""Extract the name of the corporate entity from this passage.
                    Passage:
                    {content}
                    Entity:""")

    chain = LLMChain(llm=VertexAI(model_name="text-bison", temperature=0.2),
                     prompt=prompt)

    return chain.predict(content=content).strip().upper()


def extract_tax_id_number(content):
    prompt = PromptTemplate(
        input_variables=["content"],
        template="""Extract the business number / tax identification number from this passage.
                    Passage:
                    {content}
                    Entity:""")

    chain = LLMChain(llm=VertexAI(model_name="text-bison", temperature=0.2),
                     prompt=prompt)

    return chain.predict(content=content).strip()


def extract_entity_details(content):
    prompt = PromptTemplate(
        input_variables=["content"],
        template="""What is the name of the entity, corporate registration number, date of incorporation,
                    type of entity, address, and jurisdiction in these articles of incorporation?
                    The output should be a JSON object with the following schema:
                    {{
                    "entity_name": string  // Name of the corporate entity
                    "corporation_number": string  // Corporation number of the entity (should contain numbers)
                    "formation_date": string  // Date of incorporation or formation (YYYY-MM-DD)
                    "entity_type": string // Type of entity (e.g. corporation, limited liability company)
                    "address": string // Mailing address with street, city, state/province, and zip/postal code
                    "home_jurisdiction": string // Jurisdiction of incorporation (State/Province, Country)
                    }}
                    Do not include keys if they are not present in the passage.
                    Passage:
                    {content}
                    JSON:""")

    chain = LLMChain(llm=VertexAI(model_name="text-bison", temperature=0.4,
                                  max_output_tokens=1024),
                     prompt=prompt)

    output = chain.predict(content=content)

    if output != "Not Found":
        return re.sub(r'\s+', ' ', output)
