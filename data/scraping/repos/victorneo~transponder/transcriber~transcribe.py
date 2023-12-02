import textwrap
import json
import tiktoken
import sys
import os
from dotenv import load_dotenv
from whispercpp import Whisper
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, PromptTemplate, LLMChain


load_dotenv()
OPENAI_MODEL_NAME = 'gpt-3.5-turbo-16k'


class DocumentWrapper(textwrap.TextWrapper):
    def wrap(self, text):
        split_text = text.split('\n')
        lines = [line for para in split_text for line in textwrap.TextWrapper.wrap(self, para)]
        return lines



def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def transcribe_audio_file(fname: str) -> str:
    transcript_file = fname + '.txt'

    try:
        with open(transcript_file, 'r') as f:
            transcript = f.read()
    except:
        print('Local transcript file not found')
        w = Whisper('small')
        print(f'Transcribing {fname}')
        result = w.transcribe(fname)
        text = w.extract_text(result)
        transcript = '\n'.join(text)

        with open(transcript_file, 'w') as f:
            f.write(transcript)
    else:
        print('Found local transcript file')

    return transcript


def llm_organize(fname: str, transcript: str) -> str:
    llm_output_fname = fname + '.gpt'

    try:
        with open(llm_output_fname, 'r') as f:
            output = f.read()
    except:
        print(f'Local LLM output file not found, using {OPENAI_MODEL_NAME} for organization')

        # Send to LLM for creating section headers
        template = """The following is a transcript of a podcast,
        help me add section headers into the following transcript directly in Markdown
        (I should see the original transcript in each section in nice readable paragraphs):

        {transcript}"""

        prompt_template = PromptTemplate(input_variables=['transcript'], template=template)

        llm = OpenAI(temperature=0, model_name=OPENAI_MODEL_NAME)
        output = llm(prompt_template.format(transcript=transcript))

        with open(llm_output_fname, 'w') as f:
            f.write(output)
    else:
        print('Found local LLM output file')

    return output


def main(fname: str):
    transcript = transcribe_audio_file(fname)

    # count tokens: need to be below 8k for GPT 3.5 16k
    num_tokens = num_tokens_from_string(transcript, 'gpt-3.5-turbo-16k')
    print(f'Total number of tokens: {num_tokens}')

    if num_tokens >= 8000:
        print('Unable to proceed as token exceeds what GPT 3.5 16k can handle')
        sys.exit(1)

    output = llm_organize(fname, transcript)
    # output += '\n\n## Original Transcript\n{}'.format(transcript)

    wrapper = DocumentWrapper(width=100, break_long_words=False, replace_whitespace=False)

    lines = []
    for line in wrapper.wrap(text=output):
        if line.startswith('#'):
            lines.append('\n')
        lines.append(line)

    output = '\n'.join(lines).strip()

    # Write into a markdown text file for exporting
    output_file = f'{fname}.md'

    with open(output_file, 'w') as f:
        f.write(output)

    print(f'Transcript written to {output_file}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Please provide the audio file name')
        sys.exit(1)

    main(sys.argv[1])
