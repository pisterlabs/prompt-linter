import yaml
import pandas as pd
import os
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings


def read_yaml_settings(filename):
    with open(filename, 'r') as f:
        settings = yaml.safe_load(f)
    return settings

def read_excel_to_df(file_path, sheet_name):
    return pd.read_excel(file_path, sheet_name=sheet_name)

def filter_df(df, column_name, value):
    return df[df[column_name] == value]

def rename_and_filter_columns(df, column_mapping):
    # Keep only the columns you need
    columns_to_keep = list(column_mapping.keys())
    df = df.loc[:, columns_to_keep]
    
    # Rename the columns
    df.rename(columns=column_mapping, inplace=True)
    
    return df  # This line was missing

def generate_document(row, source_name):
    filtered_row = row.dropna()
    columns_to_keep = ['Name', 'Attribute', 'Misc', 'Code', 'Tags', 'Contributor']
    filtered_row = filtered_row.loc[filtered_row.index.isin(columns_to_keep)]
    concatenated_string = ' '.join(filtered_row.astype(str))
    
    # Include unit and CO2 equivalent cost in the content
    unit = row['Unit']
    co2_equiv = row['CO2Equiv']
    concatenated_string += f' has a CO2 equivalent cost of {co2_equiv} per {unit}'
    
    metadata = {
        'source': source_name,
        'unit': row['Unit'],
        'co2_equiv': row['CO2Equiv']
    }
    
    return Document(page_content=concatenated_string, metadata=metadata)


if __name__ == '__main__':

    # to get the api key from .env
    load_dotenv()

    # Extract

    settings = read_yaml_settings('settings.yaml')
    data_file_path = settings['data_file_path']
    data_sheet = settings['data_sheet']
    
    df = read_excel_to_df(data_file_path, data_sheet)

    # Transform

    filtered_df = filter_df(df, 'Type Ligne', 'Elément')

    # Define the mapping of old column names to new column names
    column_mapping = {
        "Identifiant de l'élément": 'ElementId',
        'Nom base français': 'Name',
        'Nom attribut français': 'Attribute',
        'Nom frontière français': 'Misc',
        'Code de la catégorie': 'Code',
        'Tags français': 'Tags',
        'Unité français': 'Unit',
        'Contributeur': 'Contributor',
        'Incertitude': 'ErrorRate',
        'Total poste non décomposé': 'CO2Equiv'
    }

    # Rename and filter the columns
    final_df = rename_and_filter_columns(filtered_df, column_mapping)

    # The name of your data file for the 'source' metadata
    source_name = data_file_path.split('/')[-1]

    # Create Document objects
    documents = final_df.apply(generate_document, axis=1, args=(source_name,)).tolist()
    print("Found {} documents".format(len(documents)))

    # Load those into a faiss index
    faiss_index = FAISS.from_documents(documents, embedding=OpenAIEmbeddings())
    faiss_db = os.path.join(settings['faiss_db_path'], 'faiss_db')
    faiss_index.save_local(faiss_db)
    print('Saved faiss index to {}'.format(faiss_db))
