import json
import re
from langchain.chains import RetrievalQA
from utils.functions import find_nth, add_json_characters, add_json_characters_intro
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI


def intro_schemas(keyword, format_instructions, retriever, prompt):
    chat = ChatOpenAI(
        temperature=0,
        model_name='gpt-3.5-turbo-16k-0613'
    )

    llm = LLMChain(llm=chat, 
        prompt=prompt)

    qa = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=retriever)
    print("<----- closest")
    print(qa.run(keyword))
    closest = qa.run(keyword)
    print("<----- closest end")

    if len(closest)<300:
        return 'none'


    temp = """

    Don't forget the closing p tag. Don't add any h3 tags.
    Do not write anything about Artificial Intelligence. If anything is about artificial intelligence remove it.
    Output in html format with 'p' tags to separate paragraphs.
    Output in html format without any subheadings or h3 tags.
    Make sure to write as a blog writer NOT as the manufacturer. Don't start the intro with 'Yes'.
    Don't add any titles or forword before the introduction.
    Expand on this keyword in 2, 50 word paragraphs for my introduction for my article about "{keyword}".
    Use the context below which is based on real article summaries.
    There should be at least 3 paragraphs. It should be under 300 words.

    Use this context (real article summaries) to create the intro.
    Context: {context}

    Format the output as JSON with the following keys:
    blog_section

    {format_instructions}

    Final Checks:
    Is the entire thing shorter than 200 words? If not, shorten it.
    Are any of the paragraphs longer than 50 words? If so, break them up into smaller paragraphs.
    Is there a closing quotation mark for the JSON content? If not, add one.


    Make sure to include the closing curly brackets of the JSON.

    Section:
    """


    messages = temp.format(
        keyword=keyword,
        format_instructions=format_instructions,
        context=closest,
    )


    output_dict = llm.run(input=messages)

    print("output intro")
    print(output_dict)
    print("output intro end")
    
    output_dict = output_dict.replace("\\'","'")
    output_dict = output_dict.replace('\\"',"'")

    result = re.findall(r'{([^{]*?)}', str(output_dict))

    if len(result)>0:
        try:
            t_res = result[0].replace('“',"'")
            t_res = t_res.replace('"',"'")
            nth=find_nth(t_res, "'",3)
            nth_text = t_res[nth+1:]
            res_2 = add_json_characters_intro(nth_text)
        except:
            pass
    else:
        stripped_output = output_dict.replace("{","")
        stripped_output = stripped_output.strip()
        if stripped_output.startswith('"blog_section":'):
            t_res = stripped_output.replace('"',"'")
            t_res = t_res.replace('“',"'")
            nth=find_nth(t_res, "'",3)
            nth_text = t_res[nth+1:]
            res_2 = add_json_characters_intro(nth_text)
        else:
            test_res = '{"blog_section": "'+output_dict.replace('"',"'")
            period_index = test_res.rfind(".") + 1
            res_2 = test_res[:period_index]+'"}'
    
    if "I apologize" not in str(res_2):
        print("is not in string")
        print(res_2)
        try:
            new_response = json.loads(str(res_2), strict=False)
            new_response = new_response['blog_section']
        except:
            if res_2[-2] != '"':
                new_response = res_2[:-2]+'"'+"}"
                new_response = json.loads(str(new_response), strict=False)
                new_response = new_response['blog_section']
    else:
        new_response = res_2
    




    print("<--new res intro start")
    print(new_response)
    print("<---new res intro end")
    return new_response
