import os
import re
from IPython import embed
#from openai_model import *
from openai_model import *


def call_the_model(args, input_string):
    if args.prompt_name == '':
        prompt_string = ''
    else:
        prompt_string = open(
            os.path.join('prompts', args.prompt_name + '.txt')).read()
        if '\\n' in prompt_string:
            prompt_string = re.sub('\\\\n', '\n', prompt_string)

    if not args.do_chat:
        model = OpenaiModel(
            args.model, args.api_key, prompt_string=prompt_string)

        model_output = model.call(
            test_example_prompt=input_string,
            temperature=args.temperature,
            max_tokens=args.max_decoding_steps,
            stop=args.stop_token)

        extracted_output = model.extract_answer(model_output,
                                            args.split_start,
                                            args.split_end)
        # print(extracted_output)

    if args.do_chat:
        model = OpenaiModelChat(
            args.model, args.api_key, prompt_string=prompt_string)

        model_output = model.call(
            test_example_prompt=input_string,
            temperature=args.temperature,
            max_tokens=args.max_decoding_steps,
            stop=args.stop_token)

        extracted_output = model_output
        # print(extracted_output)
            
    # model = OpenaiModel(
    #     args.model, args.api_key, prompt_string=prompt_string)
    #
    # model_output = model.call(
    #     test_example_prompt=input_string,
    #     temperature=args.temperature,
    #     max_tokens=args.max_decoding_steps,
    #     stop=args.stop_token)

    if args.verbose:
        print(f'Model  input: {input_string}')
    # extracted_output = model.extract_answer(model_output,
    #                                         args.split_start,
    #                                         args.split_end)
    for st in ['Answer:']:
        if st in extracted_output:
            extracted_output = extracted_output.split(st)[1]
            
    if args.verbose:
        print(f'Model output: {extracted_output}')

    if args.embed:
        embed()

    return extracted_output
