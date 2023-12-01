import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import openai
import os
from dotenv import load_dotenv
import argparse

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")


# Define the generateDesc function
def generateDesc(column, examples):
    # Define the role of the system and user
    role_system = "You are an AI assistant. Given a data sample, generate a one sentence description for what the column represents (like if it's zipcode, date, etc.).  '{}'.".format(column)
    role_user = "Here is the sample data: {}. Use this to understand what the column may represent, this is not the only data but merely a sample. Start with 'Column representing xxx (eg. zipcode, date, etc.)' ".format(examples)
    
    
    # Construct the chat messages
    messages = [
        {"role": "system", "content": role_system},
        {"role": "user", "content": role_user}
    ]

    # Send a chat message to OpenAI
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
    )

    # Get the description from the response
    description = response['choices'][0]['message']['content']

    return description

def csv_to_parquet_with_metadata(csv_file, parquet_file):
    # Read csv file
    df = pd.read_csv(csv_file, delimiter=',')

    # Convert the dataframe to a PyArrow Table
    table = pa.Table.from_pandas(df, preserve_index=False)

    # Generate a schema dynamically from the DataFrame
    schema_dict = {col: {"short_name": col, "dtype": str(df[col].dtype), "examples": df[col].sample(2).tolist()} for col in df.columns}

    # Update the schema of the table
    for column, attrs in schema_dict.items():
        # Generate description
        description = generateDesc(column, attrs['examples'])

        # Update the description in the schema
        attrs['description'] = description

        # Generate new field with the updated description
        field = pa.field(column, table.column(column).type, metadata={'description': attrs['description']})

        # Replace the original column with the new field
        table = table.set_column(table.schema.get_field_index(column), field, table.column(column))

    # Write the table to a parquet file with the metadata
    pq.write_table(table, parquet_file)


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Convert CSV file to Parquet file and add metadata.')
parser.add_argument('input', type=str, help='Input CSV file name')
parser.add_argument('output', type=str, help='Output Parquet file name')

args = parser.parse_args()

# Convert CSV to Parquet with metadata
csv_to_parquet_with_metadata(args.input, args.output)
