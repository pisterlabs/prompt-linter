from langchain.prompts import PromptTemplate

JSON_FORMATTER_PROMPT_TMPL = (
    "CONSTRAINTS:\n"
    "Your job is to convert the data given to you into parsable JSON form.\n"
    "Convert the data below to JSON form.\n"
    "Your output should be in a form that can be parsed by python's json library.\n"
    "A numbered list must hold a number and have it as a string, including newlines.\n"
    "Never arbitrarily transform data. If the incoming data is in the form of a list, store it as text, not a list.\n"
    "[IMPORTANT]Only two data types are allowed in JSON: str and int.\n"
    "[IMPORTANT]List-type data is strictly prohibited. Lists should be treated as strings.\n"
    "Only return data in JSON format\n"
    "DATA:\n"
    "--------\n"
    "{data}\n"
    "--------\n"
)

JSON_FORMATTER_PROMPT = PromptTemplate(
    input_variables=["data"],
    template=JSON_FORMATTER_PROMPT_TMPL,
)