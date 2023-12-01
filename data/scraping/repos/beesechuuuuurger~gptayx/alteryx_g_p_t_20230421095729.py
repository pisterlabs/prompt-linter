# Copyright (C) 2022 Alteryx, Inc. All rights reserved.
#
# Licensed under the ALTERYX SDK AND API LICENSE AGREEMENT;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.alteryx.com/alteryx-sdk-and-api-license-agreement
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING

from ayx_python_sdk.core import (
    Anchor,
    PluginV2,
)
from ayx_python_sdk.providers.amp_provider.amp_provider_v2 import AMPProviderV2
import AlteryxPythonSDK as Sdk
import pandas as pd
import re
from langchain import LanguageChain, OpenAIAPIKeyError
from AlteryxPythonSDK import AyxEmitter


class GPTPlugin(PluginV2):
    def __init_plugin__(self):
        # Initialize instance variables
        self.langchain = None
        self.api_key = None

        self.input_column = None
        self.optional_input = None
        self.classification_attributes = []

        self.prompt_format = None
        self.output_format = None


    def initialize_output_anchor(self):
        self.output_anchor = self.output_anchor_mgr.get_output_anchor('Output')
        self.record_info_out = Sdk.RecordInfo(self.alteryx_engine)

        # Define output schema
        # You can add more output fields based on your requirements and user inputs
        self.record_info_out.add_field(
            'InputValue', Sdk.FieldType.v_wstring, 1073741823, 0)
        self.record_info_out.add_field(
            'Classification', Sdk.FieldType.v_wstring, 1073741823, 0)

        self.output_anchor.init(self.record_info_out)

    def initialize_ui_elements(self):
        self.alteryx_engine.set_init_var(self.n_tool_id, "IsMacro", "False")

        # Initialize the UI elements
        api_key = self.alteryx_engine.create_control(
            self.n_tool_id, 0, "api_key", Sdk.ControlType.text_box)
        api_key.tool_tip = "Enter your OpenAI API Key"

        input_column = self.alteryx_engine.create_control(
            self.n_tool_id, 1, "input_column", Sdk.ControlType.text_box)
        input_column.tool_tip = "Enter the name of the input column to classify"

        optional_input = self.alteryx_engine.create_control(
            self.n_tool_id, 2, "optional_input", Sdk.ControlType.text_box)
        optional_input.tool_tip = "Enter any optional input to aid in classification"

        classification_attributes = self.alteryx_engine.create_control(
            self.n_tool_id, 3, "classification_attributes", Sdk.ControlType.text_box)
        classification_attributes.tool_tip = "Enter a comma-separated list of attributes to classify"

        prompt_format = self.alteryx_engine.create_control(
            self.n_tool_id, 4, "prompt_format", Sdk.ControlType.text_box)
        prompt_format.tool_tip = "Enter the format for the prompt to be used for classification"

        output_format = self.alteryx_engine.create_control(
            self.n_tool_id, 5, "output_format", Sdk.ControlType.text_box)
        output_format.tool_tip = "Enter the output format for the classification results"

        # Add the UI elements to the plugin
        self.alteryx_engine.init_tool(self.n_tool_id, [
            api_key, input_column, optional_input, classification_attributes, prompt_format, output_format])

    def get_metadata(self) -> dict:
        return {
            'input': {
                'Input': Anchor(type_=pd.DataFrame)
            },
            'output': {
                'Output': Anchor(type_=pd.DataFrame)
            },
        }

    def submit_api_key(self, api_key):
        # Validate and set the API key for the LanguageChain object
        try:
            self.langchain = LanguageChain(api_key=api_key)
        except OpenAIAPIKeyError:
            # Handle invalid API key
            pass

    def on_input(self, input_num, data_record):
        # Retrieve input data values from the data_record
        input_value = data_record[self.record_info_in[self.input_column]].get_as_string(
        )
        optional_input_value = None
        if self.optional_input:
            optional_input_value = data_record[self.record_info_in[self.optional_input]].get_as_string(
            )

        # Process input data and store in instance variables
        if input_num == 0:
            # Process and store the input column values
            self.input_data.append(input_value)
        elif input_num == 1 and self.optional_input:
            # Process and store the optional predetermined categories
            self.optional_input_data.append(optional_input_value)

    def process_data(self, input_value: dict) -> dict:
        # Initialize an empty DataFrame to store the results
        result_df = pd.DataFrame()

        # Iterate through the stored input data
        for input_value in self.input_data:
            # Build the prompt using the input value, user-defined prompt format, and classification attributes
            prompt = self.build_prompt(input_value)

            # Handle the token count and split the prompt if necessary
            self.handle_token_count(prompt)

            # Call the LanguageChain API with the prompt and get the response
            response = self.call_langchain_api(prompt)

            # Parse the response and create a DataFrame
            response_df = self.parse_response(response)

            # Append the response DataFrame to the result DataFrame
            result_df = result_df.append(response_df, ignore_index=True)

        # Output the result DataFrame to the specified columns
        self.output_data(result_df)

    def process_optional_input_data(self):
        # Initialize an empty DataFrame to store the results
        result_df = pd.DataFrame()

        # Iterate through the stored optional input data
        for idx, optional_input_value in enumerate(self.optional_input_data):
            # Retrieve the corresponding input value
            input_value = self.input_data[idx]

            # Build the prompt using the input value, user-defined prompt format, and the predetermined category
            prompt = self.build_optional_prompt(
                input_value, optional_input_value)

            # Handle the token count and split the prompt if necessary
            self.handle_token_count(prompt)

            # Call the LanguageChain API with the prompt and get the response
            response = self.call_langchain_api(prompt)

            # Parse the response and create a DataFrame
            response_df = self.parse_response(response)

            # Append the response DataFrame to the result DataFrame
            result_df = result_df.append(response_df, ignore_index=True)

        # Output the result DataFrame to the specified columns
        self.output_data(result_df)

    def build_prompt(self):
        # Create the prompt using the user input
        prompt = f"Classify the following values based on the categories provided:\n\n"

        for value in self.input_values:
            prompt += f"- {value}\n"

        prompt += f"\nFormat the response as: 'input_value: {self.output_format}'."

        return prompt

    def build_optional_prompt(self, input_value, optional_input_value):
        # Create the prompt using the input value, user-defined prompt format, and the predetermined category
        prompt = f"{self.prompt_format} {input_value} {optional_input_value}"
        return prompt

    def handle_token_count(self, prompt):
        # Calculate the token count for the given prompt
        token_count = len(prompt.split())

        # Check if the token count exceeds the allowed limit
        if token_count > self.token_limit:
            # Split the prompt into multiple parts
            prompt_parts = self.split_prompt(prompt)

            # Process each part individually (you might need to adjust this depending on your use case)
            for part in prompt_parts:
                # Call the LanguageChain API with the split prompt part and get the response
                response = self.call_langchain_api(part)

                # Parse the response and create a DataFrame
                response_df = self.parse_response(response)

                # Output the response DataFrame to the specified columns
                self.output_data(response_df)
        else:
            # If the token count is within the allowed limit, no further action is needed
            pass

    def split_prompt(self, prompt):
        # Split the prompt into words
        words = prompt.split()

        # Initialize an empty list to store the prompt parts
        prompt_parts = []

        # Pack words into prompts while ensuring the token count does not exceed the allowed token limit
        current_part = []
        current_token_count = 0
        for word in words:
            word_token_count = len(word.split())

            # Check if adding the word would exceed the token limit
            if current_token_count + word_token_count > self.token_limit:
                # If it would, save the current part and start a new one
                prompt_parts.append(' '.join(current_part))
                current_part = [word]
                current_token_count = word_token_count
            else:
                # If not, add the word to the current part
                current_part.append(word)
                current_token_count += word_token_count

        # Add the last part to the prompt_parts list
        if current_part:
            prompt_parts.append(' '.join(current_part))

        return prompt_parts

    def call_langchain_api(self, prompt):
        # Call the LanguageChain API with the prompt and get the response
        response = self.langchain.ask(prompt)
        return response

    def parse_response(self, response):
        # Parse the response based on the output_format
        pattern = re.compile(f"input_value: {self.output_format}")

        input_values = []
        classifications = []

        for line in response.splitlines():
            match = pattern.match(line)

            if match:
                input_value, classification = match.groups()
                input_values.append(input_value.strip())
                classifications.append(classification.strip())

        # Create a DataFrame with the extracted information
        data = {'InputValue': input_values, 'Classification': classifications}
        df = pd.DataFrame(data)

        return df

    def output_data(self, df):
        # Iterate through the rows of the DataFrame
        for index, row in df.iterrows():
            # Create a new record
            record = self.record_info_out.construct_record()

            # Assign the values from the DataFrame to the corresponding output fields
            record.set_value('InputValue', row['InputValue'])
            record.set_value('Classification', row['Classification'])

            # Output the record
            self.output_anchor.push_record(record)

        # Signal the end of the data stream
        self.output_anchor.close()

    def on_complete(self):
        # Check if there is any data to output
        if not self.output_anchor.has_data():
            self.alteryx_engine.output_message(
                self.n_tool_id, Sdk.EngineMessageType.error, 'No data to output')
            return

        # Finalize the output
        self.output_anchor.output_complete()


if __name__ == "__main__":
    provider = AMPProviderV2(GPTPlugin)
    provider.run()
