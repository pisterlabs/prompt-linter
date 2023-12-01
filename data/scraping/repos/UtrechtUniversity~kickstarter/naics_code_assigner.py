'''
Module for NAICSCodeAssigner class
'''

import os
import shutil
from pathlib import Path
import pandas as pd
import openai
from utilities import retry

class NAICSCodeAssigner:

    '''
    Class for assigning a suitable NAICS code to a business idea.

    Parameters
    ----------
    openai_api_key: string
        API key from OpenAI.
    prompt: string
        Prompt to provide to ChatGPT.
    input_file: string
        Filename of the input csv file.
    model: string
        GPT model
    output_file: string
        Filename to which the results are written.
        This is the input csv file with an additional column
        containing the NAICS code.
    num_digits_naics_code: int
        Number of digits of the desired NAICS code.
    columns_business_description: list of strings
        Columns of input_filename to use to create
        the description of the business idea. Input as list of strings.
    max_retries: int
        Number of times to retry sending the prompt to OpenAI
        before giving up.
    min_wait_time: int
        Minimum time (in seconds) to wait before the next retry attempt.
    chunk_size: int
        Number of rows of input file processed at a time
    '''

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments

    def __init__(self, input_filename, separator_input_file,
                 model='gpt-3.5-turbo',
                 output_dir=None, output_filename=None, prompt_template=None,
                 year=2017, num_digits_naics_code=4,
                 columns_business_description=None,
                 max_retries=10, min_wait_time=5, chunk_size=100):

        self.input_filename = input_filename
        self.separator_input_file = separator_input_file
        self.model = model
        self.num_digits_naics_code = num_digits_naics_code
        self.max_retries = max_retries
        self.min_wait_time = min_wait_time
        self.chunk_size = chunk_size

        if output_dir is not None:
            self.output_dir = output_dir
        else:
            self.output_dir = './data'

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        if output_filename is not None:
            self.output_filename = self.output_dir + '/' + self.output_filename
        else:
            self.output_filename = self.output_dir + \
                '/' + f'data_with_{year}_naics_code.csv'

        if prompt_template is not None:
            self.prompt_template = prompt_template
        else:
            self.prompt_template = ''' \
            Assign the closest 4-digit 2017 NAICS code to this business idea. \
            List only the 4-digit 2017 NAICS code, and no other text. \
            For example, for the 2017 NAICS category \
            'Motion Picture and Video Industries', print: \
            '5121'\
            '''

        if columns_business_description is not None:
            self.columns_business_description = columns_business_description
        else:
            self.columns_business_description = ['title', 'blurb', 'category', 'subcategory']

        self.progress_log = 'progress.txt'

    def obtain_naics_codes(self):

        '''
        Assign NAICS codes to all entries of the input file,
        and write the results incrementally, chunk by chunk,
        to the output file.
        '''

        # Ensure that the header is written only if
        #the output file is empty or non-existent
        if not os.path.exists(self.output_filename):
            write_header = True
        elif os.path.getsize(self.output_filename)==0:
            write_header = True
        else:
            write_header = False

        # Check progress from the log
        try:
            with open(self.progress_log, 'r', encoding='utf-8') as file:
                start_chunk = int(file.read())
        except (FileNotFoundError, ValueError):
            start_chunk = 0

        # Read the csv file in chunks
        chunks = pd.read_csv(self.input_filename, \
                             chunksize = self.chunk_size, \
                             sep = self.separator_input_file)

        # Process one chunk at a time
        for i, chunk in enumerate(chunks, start=1):

            # If a chunk has already been processed, skip it
            if i <= start_chunk:
                continue

            print(f'Processing chunk {i} ...')

            # Remove NaN values
            chunk.fillna('', inplace=True)
            chunk = self.obtain_naics_codes_chunk(chunk)

            if i>1:
                # Make backup copy of output file before appending to it
                shutil.copy(self.output_filename, \
                            self.output_filename.rsplit('.', 1)[0] \
                            + '_backup1.' \
                            + self.output_filename.rsplit('.',1)[1])

            # Append results from this chunk to output file
            chunk.to_csv(self.output_filename, mode='a', \
                         header=write_header, sep = '\t', index=False)

            # Make backup copy of output file after appending to it
            shutil.copy(self.output_filename, \
                        self.output_filename.rsplit('.', 1)[0] \
                        + '_backup2.' + self.output_filename.rsplit('.',1)[1])

            # Write header only once
            if write_header:
                write_header = False

            # Update progress log
            with open(self.progress_log, 'w', encoding='utf-8') as file:
                file.write(str(i))

            print(f'Finished processing chunk {i}.\n')

    def obtain_naics_codes_chunk(self, chunk):

        '''
        Parameters
        ----------
        chunk : dataframe
            Input dataframe

        Returns
        -------
        chunk : dataframe
            Output dataframe, with additional columns appended for
            NAICS codes and number of tokens used.

        '''

        naics_codes = []
        num_prompt_tokens = []
        num_completion_tokens = []

        for _, row in chunk.iterrows():

            response = self.assign_naics_code(row)

            if response is not None:
                naics_code = response.choices[0].message['content']
                prompt_tokens = response.usage['prompt_tokens']
                completion_tokens = response.usage['completion_tokens']
            else:
                naics_code = None
                prompt_tokens = None
                completion_tokens = None

            naics_codes.append(naics_code)
            num_prompt_tokens.append(prompt_tokens)
            num_completion_tokens.append(completion_tokens)

        chunk['naics code'] = naics_codes
        chunk['input tokens'] = num_prompt_tokens
        chunk['output tokens'] = num_completion_tokens

        return chunk

    def assign_naics_code(self, row):

        '''
        Parameters
        ----------
        row : dataframe
            Row of input file being processed.

        Returns
        -------
        response : dict
            Dictionary containing NAICS code and information about
            number of tokens used.

        '''

        prompt = self.create_prompt(row)
        response = retry(self.get_completion, self.max_retries, \
                         self.min_wait_time, prompt, self.model)

        return response

    def get_completion(self, prompt, model='gpt-3.5-turbo'):

        '''
        Parameters
        ----------
        prompt : str
            Prompt to be provided to ChatGPT.

        model : str
            GPT model being used. The default is 'gpt-3.5-turbo'.

        Returns
        -------
        response : dictionary
            Output from ChatGPT.

        '''

        messages = [{'role': 'user', 'content': prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0,
            )

        return response

    def create_prompt(self, row):

        '''
        Parameters
        ----------
        row : dataframe
            Row of input file being processed.

        Returns
        -------
        prompt : str
            Prompt corresponding to this row,
            containing the text from the data in this row.
        '''

        description = self.create_business_description(row, self.columns_business_description)
        prompt = f'''{description}. {self.prompt_template}'''

        return prompt

    def create_business_description(self, row, columns_business_description):

        '''
        Parameters
        ----------
        row : dataframe
            Row of input file being processed.

        columns_business_description : list of str
            List of columns from input file to be used in the prompt.

        Returns
        -------
        description : str
            Description of the business idea, formed by concatenating
            the columns of columns_business_description.

        '''

        description = '. '.join(row[columns_business_description])

        return description
