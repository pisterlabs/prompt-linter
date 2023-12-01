import openai
import random

# generates continuation but throws out all delimiter-separated strings that match
# any string in the prompt or previously generated strings verbatim
#
# if batch size not specified, generates with delimiter as end sequence (e.g. one line)
def generate_no_verbatim(chars_per_batch=None, delimiter='\n', max_attempts=None, prepend_string='', scramble=False, **kwargs):
    prompt_lines = kwargs['prompt'].split(delimiter)
    max_attempts = max_attempts if max_attempts else 20
    unique_response_lines = []
    repeated_response_lines = []
    attempts = 0
    while attempts < max_attempts:
        if scramble:
            random.shuffle(prompt_lines)
        prompt = delimiter.join(prompt_lines)
        if prompt[-1] is not delimiter:
            prompt = prompt + delimiter
        prompt_length = (5000 - chars_per_batch) if chars_per_batch else 5000
        prompt_length -= len(prepend_string)
        prompt = prepend_string + prompt[-prompt_length:]
        rsp = openai.Completion.create(
            engine=kwargs['engine'] if 'engine' in kwargs else 'curie',
            prompt=prompt,
            temperature=kwargs['temperature'] if 'temperature' in kwargs else 1,
            max_tokens=chars_per_batch if chars_per_batch else 500,
            echo=False,
            top_p=1,
            n=1,
            stop=None if chars_per_batch else delimiter)
        attempts += 1
        rsp_text = rsp.choices[0]['text']
        rsp_text_lines = rsp_text.split(delimiter)
        if chars_per_batch:
            rsp_text_lines.pop()
        for line in rsp_text_lines:
            if not line.isspace():
                if line not in prompt_lines:
                    prompt_lines.append(line)
                    unique_response_lines.append(line)
                else:
                    repeated_response_lines.append(line)

    return unique_response_lines, repeated_response_lines