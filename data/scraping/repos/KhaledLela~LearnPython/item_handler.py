import json
import os
import re
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser

from LearnPython.AI.app.item.resource_item_pydantic import ResourceItem
from LearnPython.AI.prompt.prompt_templates import item_title_prompt, item_regenerate_prompt, item_translate_prompt, \
    item_follow_up_prompt, item_create_prompt


def item_prompt(event):
    item: dict = create_item(event)
    response = json.dumps(item)
    return {
        'statusCode': 200,
        'body': response
    }


def create_item(params):
    """
      Load and run create item chain
    """
    llm = ChatOpenAI(model_name="gpt-4",
                     openai_api_key=os.getenv("OPENAI_API_KEY"))
    # Set up a parser + inject instructions into the prompt template.
    parser = PydanticOutputParser(pydantic_object=ResourceItem)
    prompt_template = map_prompt(parser, params)
    chain = LLMChain(llm=llm, verbose=True, prompt=prompt_template)
    result = chain.generate(input_list=[params])
    print(result)

    token_usage = result.llm_output['token_usage']
    output = result.generations[0][0].text
    try:
        item = parser.parse(output).dict()
    except Exception as ex:
        print(ex)
        match = re.search(r'"text": "(.*?)",', output, re.DOTALL)
        item = {'text': match.group(1)} if match else {'text': output}
    id = str(result.run[0].run_id)

    return {'prompt_uuid': id} | item | token_usage


def map_prompt(parser, params):
    action_type = params.get('action_type')
    source = params.get('source')

    if action_type == 'merge':
        return item_title_prompt(parser)
    elif action_type == 'regenerate':
        return item_regenerate_prompt(parser)
    elif action_type == 'translate':
        return item_translate_prompt(parser)
    elif source != "":
        return item_follow_up_prompt(parser)
    else:
        return item_create_prompt(parser)
