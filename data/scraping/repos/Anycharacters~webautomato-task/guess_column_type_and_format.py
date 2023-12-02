from langchain.prompts import PromptTemplate

def make_prompt_for_column_name(needle_column_heading, needle_column_data_sample, haystack_column_headings, haystack_column_data_samples):

    prompt_template = PromptTemplate.from_template(
        "\nYou are an intelligent assistant. Memorize that we have columns \n"+
        "\n".join([ f" column name is '{heading}' and data type for the column could be inferred from the sample  {sample}" for heading, sample in zip(haystack_column_headings, haystack_column_data_samples)])
        +"\n\n Now you need to find out which of the above column names is the most relevant for the column with name"
        +"\n '{needle_column_heading}' and the data type for the column could be inferred from sample [{needle_column_data_sample}]"
        +"\nAfter you have found the most relevant column name for '{needle_column_heading}', reply only with that column name in one word, do not write explanations."
        +"\nMake sure that data type for the column '{needle_column_heading}' is the same as for the column you have found."
    )
    return prompt_template.format(needle_column_heading=needle_column_heading,
                                    needle_column_data_sample=needle_column_data_sample)

def make_prompt_for_format_change(old_format,new_format):
    return f"\nYou are an intelligent assistant. We have the values old_format='{old_format}' and new_format='{new_format}' which means '{old_format}' needs to be converted to the target format'{new_format}'."+     "We have python function \n"+"""def format_number(number):
    pattern = r"(\w{2})(\d+)"
    replacement = r"\1-\2"
    formatted_number = re.sub(pattern, replacement, number)
    return formatted_number""" +    "\n\nSo, the language model needs to change 2 variables 'pattern' and 'replacement' in 'format_number'" +    "in python code using regex import re\nDon't write an explanation.\n" +    """Example:
        User{ convert the format "SD12984" to "SD-12984" }
        AI{'r"(\w{2})(\d+)"'#'r"\1-\2"'}
    """ +    "Now you need to find 2 variables 'pattern' and 'replacement' in the request\n" +    "User{ convert the value old_format='" + old_format + "' to new_format='" + new_format + "'}" +    '\nAI{"your answer on pattern here"#"your answer on replacement here"}' +    "\nIn your reply, omit the part '" +    "User{ convert the value old_format='" + old_format + "' to new_format='" + new_format + "'}AI" +    ".\n\nJust return dictionary " +    '\n{"your answer on pattern here"#"your answer on replacement here"}'

