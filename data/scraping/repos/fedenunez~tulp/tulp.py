#!/usr/bin/python3
import openai
import sys
import os
import math
from . import tulpargs
from . import tulplogger
from . import tulpconfig
from . import version
from . import TulpOutputFileWriter

log = tulplogger.Logger()
config = tulpconfig.TulipConfig()
args = tulpargs.TulpArgs().get()



# Helper functions

## cleanup_output: clenaup output, removing artifacts
def cleanup_output(output):
    olines = output.splitlines()
    # gpt-3.5 usually adds unneeded markdown codeblock wrappers, try to remove them
    if len(olines) > 2:
        if olines[0].startswith("```") and olines[-1] == "```":
            log.debug("markdown codeblock wrapping detected and stripped!")
            return "\n".join(olines[1:-1])
    return output


## block_exists(blocks_dict, key):  test if a KEY exist and is not empty
def block_exists(blocks_dict, key):
    return key in blocks_dict and len(blocks_dict[key]["content"].strip()) > 0

## VALID_BLOCKS: define the valid answer blocks  blocks
VALID_BLOCKS=["(#output)","(#thoughts)","(#inner_message)","(#error)","(#context)","(#comment)","(#end)"]
## parse_response)response_text): parse a gpt response, returning a dict with each response section 
def parse_response(response_text):
    blocks_dict={}
    lines = response_text.splitlines()
    # (#end) is not specified by us, but sometimes gpt-3.5 wrote it so we just parse it so we can keep it out

    # parse blocks:
    parsingBlock=None
    for line in lines:
        splitted_line = line.split(" ")
        key = None
        if (len(splitted_line) >= 1):
            key = splitted_line[0]
            if (len(splitted_line) > 2):
                outputType = " ".join(splitted_line[1:])
            else:
                outputType = None
        if (key in VALID_BLOCKS):
            parsingBlock=key
            if parsingBlock not in blocks_dict:
               blocks_dict[parsingBlock]={}
               blocks_dict[parsingBlock]["type"]=outputType
               blocks_dict[parsingBlock]["content"] = ""


        else:
            if parsingBlock:
                blocks_dict[parsingBlock]["content"] += line + "\n"
            else:
                if config.model == "gpt-3.5-turbo": 
                    log.error("""
Unknown error while processing: this is usually related to gpt not honoring our
output format, please try again and try to be more specific, you can also try
with a different model (e.g., TULP_MODEL=gpt-4 tulp ...)""")
                else:
                    log.error("""
Unknown error while processing: this is usually related to gpt not honoring
our output format, please try again and try to be more specific in your
request. You can also try to enable DEBUG log to inspect the raw answer (e.g.,
TULP_LOG_LEVEL=DEBUG tulp ...)""") 
                log.debug(f"ERROR: Invalid answer format: =====\n {response_text} \n=====")
                sys.exit(2)
    return blocks_dict


## pre_process_raw_input(input_text):  split all the input and create the raw_input_chunks with chunks of text ready to be send 
def pre_process_raw_input(input_text):
    raw_input_chunks=[]
    # Split input text into chunks to fit within max chars window
    max_chars = config.max_chars  # Maximum number of chars that we will send to GPT
    if len(input_text) > max_chars:
        warnMsg = f"""
Input is too large ({len(input_text)} characters). Typically, tulp does not handle inputs
larger than 5000 characters well. Regardless, tulp will divide the input into
chunks of fewer than {max_chars} characters and attempt to process all the input.

Please be aware that the quality of the final result may vary depending on the
task. Tasks that are line-based and do not require context will work great,
while tasks that require an overall view of the document may fail miserably.

You may adjust the TULP_MAX_CHARS environment variable to control the size of
the processing chunks, which may improve the results.

"""
        if config.model != "gpt-4":
           warnMsg = warnMsg + """You can also try to force the use of the gpt-4 model (TULP_MODEL=gpt-4), which
usually improves the quality of the result.
"""
        log.warning(warnMsg)

    # try to split it in lines of less than max_size
    compressed_lines = [""]

    input_lines = input_text.splitlines()

    for iline in input_lines:
        compresed_index = len(compressed_lines) - 1
        clen = len(compressed_lines[compresed_index]) 
        if clen + len(iline) < max_chars:
            compressed_lines[compresed_index] += iline + "\n"
        else:
            if clen != 0:
                if (len(iline) < max_chars):
                    compressed_lines.append(iline + "\n")
                else:
                    for i in range(0,math.floor(len(iline)/max_chars)+1):
                        input_chunk = iline[i*max_chars:(i+1)*max_chars] 
                        compressed_lines.append(input_chunk)
                    compressed_lines[compresed_index] += "\n"

    for line in compressed_lines:
        raw_input_chunks.append(line)

    return raw_input_chunks


def processExecutionRequest(promptFactory, user_request, raw_input_chunks=None):
    retries = 0
    max_retries = 5
    if (not raw_input_chunks):
        raw_input_chunks = [""]
    requestMessages = promptFactory.getMessages(user_request, raw_input_chunks[0], len(raw_input_chunks))
    while retries < max_retries:
        for req in requestMessages:
            log.debug(f"REQ: {req}")
        log.debug(f"Sending the request to OpenAI...")
        response = openai.ChatCompletion.create(
            model=config.model,
            messages=requestMessages,
            temperature=0
        )
        log.debug(f"ANS: {response}")
        response_text = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason
        blocks_dict = parse_response(response_text)
        if block_exists(blocks_dict,"(#comment)"):
            log.info(blocks_dict["(#comment)"]["content"])
        if block_exists(blocks_dict,"(#output)"):
            oType = ""
            generatedCode = cleanup_output(blocks_dict["(#output)"]["content"])
            log.debug(f"The generated code:\n{generatedCode}")
            from . import executePython
            input_text="".join(raw_input_chunks)
            if args.w:
                ok, filename = TulpOutputFileWriter.TulpOutputFileWriter().write_to_file(args.w, generatedCode)
                if ok: 
                    log.info(f"Wrote created program at: {filename}")
                else:
                    log.error(f"Error while writing created code: {filename}")
            log.info(f"Executing generated code...")
            boutput, berror, ecode  = executePython.execute_python_code(generatedCode, input_text)
            log.debug(f"Execution results: {boutput}, {berror}, {ecode}")
            if (ecode != 0 and berror.find("Traceback") != -1 ):
                retries += 1
                log.warning(f"Error while executing the code, I will try to fix it!")
                requestMessages = promptFactory.getMessages(user_request, raw_input_chunks[0], len(raw_input_chunks))
                requestMessages.append(response.choices[0].message)
                requestMessages.append({"role": "user","content": f"The execution of the program failed with error:\n{berror}\n\nPlease try to write a new (#output) that fixes the error"})
            else:
                break;
    if retries == max_retries:
        log.error("while executing generated code, max retries reached, giving up.")
    else:
        if (ecode != 0):
            log.error(f"Error executing the program:\n{berror}")
        else:
            log.info("Code was executed correctly.")

    print(boutput)
    return ecode

def processRequest(promptFactory,user_request, raw_input_chunks=None):
    if not raw_input_chunks:
        raw_input_chunks = [""] 
    for i in range(0,len(raw_input_chunks)):
        if (len(raw_input_chunks) > 1):
            log.info(f"Processing {i+1} of {len(raw_input_chunks)}...")
        else:
            log.info(f"Processing...")
        finish_reason = ""
        response_text = ""
        raw_input_chunk = raw_input_chunks[i]
        if user_request:
            requestMessages = promptFactory.getMessages(user_request, raw_input_chunk , len(raw_input_chunks), i+1)
            for req in requestMessages:
                log.debug(f"REQ: {req}")
            log.debug(f"Sending the request to OpenAI...")
            response = openai.ChatCompletion.create(
                model=config.model,
                messages=requestMessages,
                temperature=0
            )
            log.debug(f"ANS: {response}")
            response_text += response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason


        blocks_dict = parse_response(response_text)

        if block_exists(blocks_dict,"(#error)"):
            log.error("Error: Couldn't process your request:")
            log.error(blocks_dict["(#error)"]["content"])
            sys.exit(1)
        else:
            valid_answer = False

            if block_exists(blocks_dict,"(#inner_message)"):
                log.debug("(#inner_message) found!")

            if block_exists(blocks_dict,"(#output)"):
                valid_answer = True
                oType = ""
                if blocks_dict["(#output)"]["type"]:
                    oType = f'(type: {blocks_dict["(#output)"]["type"]})'
                log.info(f"Writting generated output {oType}") 
                print(cleanup_output(blocks_dict["(#output)"]["content"]))

            if block_exists(blocks_dict,"(#comment)"):
                valid_answer = True
                log.info(blocks_dict["(#comment)"]["content"])

            if block_exists(blocks_dict,"(#context)"):
                prev_context = blocks_dict["(#context)"]
            else:
                prev_context = None

            if not valid_answer:
                log.error("Unknown error while processing, try with a different request, model response:")
                log.error(response_text)
                sys.exit(2)

        if finish_reason == "length":
            errorMsg = f"""Token limit exceeded:
GPT could not finish your response, the answer depleted the chatGPT token
limit. In order to overcome this error you may try to use a smaller MAX_CHARS
(currently ={config.max_chars}), using a different model or improving your
instructions.
"""

            log.error(errorMsg)
            sys.exit(2)

def run():
    log.debug(f"Running tulp v{version.VERSION} using model: {config.model}")
    openai_key = config.openai_api_key
    if not openai_key:
        log.error(f'OpenAI API key not found. Please set the TULP_OPENAI_API_KEY environment variable or add it to {tulpconfig.CONFIG_FILE}')
        log.error(f"If you don't have one, please create one at: https://platform.openai.com/account/api-keys")
        sys.exit(1)


    openai.api_key = openai_key

    # If input is available on stdin, read it
    input_text = ""
    if not sys.stdin.isatty():
        input_text = sys.stdin.read().strip()

    user_request=None
    if not args.request and not input_text:
        user_request = input("Enter your request: ").strip()
    elif args.request and not input_text:
        user_request = args.request
    elif not args.request and input_text:
        user_request = "Summarize the input"
    elif args.request and input_text:
        user_request = args.request

    raw_input_chunks = pre_process_raw_input(input_text)

    if input_text and user_request:
        # A filtering request:
        if (args.x):
            from . import createFilteringProgramPrompt
            sys.exit(processExecutionRequest(createFilteringProgramPrompt, user_request, raw_input_chunks))
        else:
            from . import filteringPrompt
            sys.exit(processRequest(filteringPrompt, user_request, raw_input_chunks))
    else:
        # A request
        if (args.x):
            from . import createProgramPrompt
            sys.exit(processExecutionRequest(createProgramPrompt, user_request))
        else:
            from . import requestPrompt
            sys.exit(processRequest(requestPrompt, user_request))



if __name__ == "__main__":
    run()
