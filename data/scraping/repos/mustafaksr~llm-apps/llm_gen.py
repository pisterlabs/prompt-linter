import re
import json
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import ChatPromptTemplate

from langchain.chat_models import ChatVertexAI
chat = ChatVertexAI( temperature = 1.0, max_output_tokens = 1024,
                    top_p = 0.8, top_k = 40)

def is_string_closed_correctly(string):
    pattern = r'\".*?(?<!\\)\"'  # Regular expression pattern for matching quoted strings

    matches = re.findall(pattern, string)
    if len(matches) % 2 == 0:
        return True  # Even number of matches indicates correct closing
    else:
        return False  # Odd number of matches indicates incorrect closing

def fn_pre(text):
    filtered_text = re.sub(r'[\t\n]', '', text)
    filtered_text = re.sub(r'```json', '', filtered_text)
    filtered_text = re.sub(r'```', '', filtered_text)
    match2 = re.search(r'"}', filtered_text)
    if is_string_closed_correctly(filtered_text) and match2 :
        pass
    else:
        if not match2:
            filtered_text +='"'
        else:pass
    
    match = re.search(r'}', filtered_text)
    if match :
        pass
    else:
        filtered_text +='}' #close if missed

    match = re.search(r",}", filtered_text)
    if match:
        filtered_text = re.sub(r',}', '', filtered_text)
        filtered_text +='}'
    else:
        pass
  
    split = text.split('"back_card":')[1]
    split0 = text.split('"back_card":')[0]
    if split.count('\"') > 2:
        filtered_split = re.sub(r'"([^"]+)"', lambda match: match.group(1) if len(match.group(1).split()) > 1 else match.group(0), split)
        filtered_split='"'+re.sub(r'}','"}',filtered_split)
        filtered_text = split0 + """"back_card":"""+filtered_split
        filtered_text = re.sub(r'```json', '', filtered_text)
        filtered_text = re.sub(r'```', '', filtered_text)
        filtered_text = re.sub(r'[\t\n]', '', filtered_text)
    return filtered_text

def make_llm_cards(user_prompt="can you create flashcards as related to something?",num=6):
    previous_Qs=[]
    results = []
    for i in range(num):
        context = f"""
        0.You are a flashcard creator and your task is to create flashcards in plain text format without breaking lines.

        1.Generate new questions that are distinct from each other. Avoid duplicating questions and do not include any of the previously generated front card questions, which were as follows: {str(previous_Qs)}.

        2.Create a single flashcard with a front side (question) and a back side (answer), using new questions.

        3.Ensure that both the front card and the back card text are no longer than 200 characters. Each time, provide only one new question and answer pair, without multiple pairs.

        4.When writing the answer on the back of the card as a string, do not include a comma (",") at the end, as it may cause JSON parsing errors.

        5.Even if the input text requests multiple flashcards, you should generate only one flashcard item. Avoid using multiple curly brackets and do not include new lines or tabs (\n and \t) within the curly brackets.

        6.For example, when using strings, pay attention to colons and double quotation marks. Write them in the following format: "Leading AI textbooks define the field as the study of intelligent agents: any device that perceives its environment and takes actions to maximize its chances of successfully achieving its goals." (Use a single quotation mark and avoid multiple quotation marks and colons, as they are incorrect in this context.)
        
        """
        
        
        _prompt = f"""\ 
         {user_prompt} \ 
        """
        _template =  f""" rules: {context}""" + """\
        
        generate questions related following input text accordingly rules: {text}
        
        {format_instructions} \ 
        """ 
        prompt = ChatPromptTemplate.from_template(template=_template)
        
        
        front_schema = ResponseSchema(name="front_card",
                                     description="Front of flashcard related to question. consider as python string.")
        back_schema = ResponseSchema(name="back_card",
                                     description="Back of flashcard related to answer. consider as python string." )
        response_schemas = [front_schema,
                            back_schema]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()
        messages = prompt.format_messages(text=_prompt,
                                                    format_instructions=format_instructions)
        response = chat(messages)
        output_dict = None
        try:
             output_dict = output_parser.parse(response.content)
        except:
            output_dict = None
            pass
        if output_dict is None:
            try:
                filtered_text = response.content
                # Remove \t and \n characters using regular expressions
                for i in range(3):
                    filtered_text= fn_pre(filtered_text)
                    
                output_dict = json.loads(filtered_text)
            except:
                    output_dict = {"front_card":"failed to parse",
                                  "back_card":response.content}
                
        results.append(output_dict)
        previous_Qs.append(output_dict["front_card"])
    
    front_text = [x["front_card"] for x in results]
    back_text =  [x["back_card"] for x in results]
    
   
    return front_text , back_text
