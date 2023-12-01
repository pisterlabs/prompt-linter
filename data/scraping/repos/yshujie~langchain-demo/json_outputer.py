from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

def output():
    # 初始化 llm
    llm = OpenAI(model_name="text-davinci-003")
    
    # 定义输出 schema
    response_schema = [
        ResponseSchema(name="bad_string", description="This is a poorly formatted user input string"),
        ResponseSchema(name="good_string", description="This is your response, a reformated response")
    ]
    
    # 初始化解析器
    output_parser = StructuredOutputParser.from_response_schemas(response_schema)
    
    # 生成的格式提示符
    format_instructions = output_parser.get_format_instructions()
    
    template = """
    You will be given a poorly formatted string from a user.
    Reformat it and make sure all the words are spelled correctly.
    
    {format_instructions}
    
    % USER INPUT
    {user_input}
    
    YOUR RESPONSE:
    """
    
    # 将我们的格式描述嵌入到 prompt 中，告诉 llm 我们需要他输出什么样格式的内容
    prompt = PromptTemplate(
        input_variables=['user_input'],
        partial_variables={'format_instructions': format_instructions},
        template=template
    )
    
    prompt_value = prompt.format(user_input="welcome to califonya!")
    llm_output = llm(prompt_value)
    
    # 使用解析器进行解析生成的内容
    return output_parser.parse(llm_output)