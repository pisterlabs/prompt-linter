import main
import json
import re
from langchain.prompts import PromptTemplate
from langchain.llms import VertexAI
from langchain.chains import LLMChain


def Parser(sorted_files):
    """
    Extracts share class details from the sorted pages of minute book.

    Args:
        sorted_files (list): A list of tuples where each tuple contains the page number and file name of a sorted file.

    Returns:
        A list of dictionaries where each dictionary represents a share class and includes its name, voting rights,
        votes per share, limit for number of shares, number of shares authorized, and share restrictions. Each share class
        object also includes the URL of the source document page where the share class details were extracted from.
    """

    share_classes = [{}]
    share_class_content = ""
    share_class_token_count = 0
    share_class_max_token_limit = 2560
    extracting_share_classes = False

    file_count = len(sorted_files)
    for file in sorted_files:
        page_number, file_name = file
        content = main.get_page(file_name)
        lowercase_content = content.lower()

        #  "share_classes": array, // One or more share classes with children properties for name, voting rights, votes per share, limit for number of shares, number of shares authorized, and share restrictions
        if "authorized to issue" in lowercase_content and "class" in lowercase_content:
            extracting_share_classes = True

        if extracting_share_classes is True:
            share_class_tokens = share_class_token_count + main.num_tokens_from_string(content)

            if share_class_tokens < share_class_max_token_limit:
                share_class_token_count += share_class_tokens
                share_class_content += content

            if share_class_tokens >= share_class_max_token_limit or page_number == file_count:
                output = extract_share_classes(share_class_content)
                try:
                    share_classes = json.loads(output)
                    share_classes.append({'provenance': main.get_url(file_name)})
                    share_classes.append(share_classes)

                    for share_class in share_classes:
                        if not bool(share_class):
                            share_classes.remove(share_class)
                except json.decoder.JSONDecodeError:
                    pass

                extracting_share_classes = False
                share_class_content = ""
                share_class_token_count = 0
                share_class_tokens = 0

    return share_classes


# The following function uses a large language model to perform question & answer-style extraction from a minute book


def extract_share_classes(content):
    prompt = PromptTemplate(
        input_variables=["content"],
        template="""What share classes is the corporation authorized to issue? Output JSON
                    objects that conform to the following schema:
                    {{
                        {{
                        "share_class": string  // Name of class of shares (example: Class A, Class B or Common, Preferred)
                        "voting_rights": string  // Yes or no
                        "votes_per_share": string // Number of votes per share
                        "notes": string  // Summarize rights, privileges, restrictions, and conditions
                        }},
                        // Repeat for each share class found
                    }}
                    Passage:
                    {content}
                    Share Classes JSON:""")

    chain = LLMChain(llm=VertexAI(model_name="text-bison", temperature=0.5,
                                  max_output_tokens=1024),
                     prompt=prompt)

    output = chain.predict(content=content)
    return re.sub(r'\s+', ' ', output)
