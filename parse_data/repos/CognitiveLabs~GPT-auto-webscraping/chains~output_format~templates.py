from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, PromptTemplate

# prompt templates
system_template_output_format = PromptTemplate(
    input_variables = ['html_content'],
    template='''You are a helpful assitant that helps people extract JSON information from HTML content.

    The input is a HTML content. 

    The expected output is a JSON with a relevant information in the following html: {html_content}

    Try to extract as much information as possible. Including images, links, etc.

    The assitant answer should ONLY contain the JSON information without any aditional word or character.

    The JSON output must have 1 depth level as much.

    The expected output format is an array of objects.
    
    ''')

human_template_output_format = PromptTemplate(
    input_variables = ['html_content'], 
    template='this is the html content: {html_content}'
)

# chat prompts objects
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template_output_format.template)
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template_output_format.template)
output_format_chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
