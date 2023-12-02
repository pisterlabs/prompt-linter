import main
import json
import re
from langchain.prompts import PromptTemplate
from langchain.llms import VertexAI
from langchain.chains import LLMChain


def Parser(sorted_files):
    """
    Extracts details of appointed officers from the sorted pages of minute book.

    Args:
        sorted_files (list): A list of tuples where each tuple contains the page number and file name of a sorted file.

    Returns:
        A list of dictionaries where each dictionary represents an appointed officer and includes their full name, appointment
        date, address, title, and URL of the source document page where the details were extracted from.
    """

    elected_officers = [{}]
    election_of_officer_content = ""
    election_of_officer_provenance = []
    election_of_officer_token_count = 0
    election_of_officer_max_token_limit = 1024
    extracting_election_of_officer = False

    file_count = len(sorted_files)
    for file in sorted_files:
        page_number, file_name = file
        content = main.get_page(file_name)
        lowercase_content = content.lower()
        parsed_this_page = False

        #  "officers": array, // One or more officers of a corporation, with children properties for their full name, election date, address, and title
        if "officer" in lowercase_content and "register" in lowercase_content:
            extracting_election_of_officer = True

        if extracting_election_of_officer is True:
            election_of_officer_tokens = election_of_officer_token_count + main.num_tokens_from_string(content)
            election_of_officer_provenance.append(main.get_url(file_name))

            if election_of_officer_tokens < election_of_officer_max_token_limit:
                election_of_officer_token_count += election_of_officer_tokens
                election_of_officer_content += content
                parsed_this_page = True

            if election_of_officer_tokens >= election_of_officer_max_token_limit or page_number == file_count:
                if parsed_this_page:
                    output = extract_election_of_officers(election_of_officer_content)
                else:
                    output = extract_election_of_officers(content)

                if output is not None:
                    try:
                        output = json.loads(output)
                        found_officers = []
                        for officer in elected_officers:
                            if not bool(officer):
                                elected_officers.remove(officer)

                            for item in output:
                                if "officer_name" in officer and officer['officer_name'] == item['officer_name']:
                                    officer['officer_name'] = item['officer_name'].title()
                                    officer['date_appointed'] = item['date_appointed']
                                    officer['date_retired'] = item['date_retired']
                                    officer['position_held'] = item['position_held']
                                    if "address" in item and isinstance(item['address'], str):
                                        officer['address'] = re.sub(r'\s+', ' ', item['address']).strip()
                                    try:
                                        officer['provenance'] = item['provenance']
                                    except KeyError as e:
                                        print(e)       
                                else:
                                    item['officer_name'] = item['officer_name'].title()
                                    item['address'] = main.extract_address_for_person(person=item['officer_name'], sorted_files=sorted_files)
                                    item['provenance'] = election_of_officer_provenance
                                    found_officers.append(item)

                        for officer in found_officers:
                            if not any(d['officer_name'] == officer['officer_name'] for d in elected_officers):
                                elected_officers.append(officer)

                    except json.decoder.JSONDecodeError:
                        pass

                extracting_election_of_officer = False
                election_of_officer_content = ""
                election_of_officer_token_count = 0
                election_of_officer_provenance = []
                election_of_officer_tokens = 0

    return elected_officers


# The following function uses a large language model to perform question & answer-style extraction from a minute book


def extract_election_of_officers(content):
    prompt = PromptTemplate(
        input_variables=["content"],
        template="""List the names of the officers of the corporation, the date they were elected,
                    and the date they retired (if not a current officer). The output should be a
                    JSON object with one or more children having the following schema:
                    {{
                    "officer_name": string  // Name of the elected officer
                    "date_appointed": string  // Formatted date (YYYY-MM-DD) of the appointed date
                    "date_retired": string  // Formatted date (YYYY-MM-DD) of the retired date
                    "position_held": string // Position held by the elected officer
                    "address": string // Address of the elected officer
                    }}
                    If the passage does not mention names of officers, output [].
                    Passage:
                    {content}
                    Officers JSON:""")

    chain = LLMChain(llm=VertexAI(model_name="text-bison", temperature=0.2,
                                  max_output_tokens=1024),
                     prompt=prompt)

    output = chain.predict(content=content).strip()

    if output != "[]":
        return output
