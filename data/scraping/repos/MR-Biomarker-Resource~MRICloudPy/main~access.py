import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI

# Renames specific subject ID
def rename_subject(self, old: str, new: str):
    if (old not in self.df['ID']):
        # Invalid ID error
        print(str(self.rename_subject.__name__) + ": Invalid old ID: " + "\'" + str(old) + "\'")
    df_id = self.df['ID'].replace(old, new)
    self.df['ID'] = df_id
    return self.df

def get_data(self):
    return self.df

# Return list of all unique IDs in a dataframe
def get_id(self):
    ids = self.df['ID'].unique()
    return ids

# Converts dataframe from long to wide form
def long_to_wide(self):

    df = self.df.copy()
    print(str(self.long_to_wide.__name__) + ": Converting...")

    # Eliminate duplicates by appending type and level data to object strings
    for i, row in df.iterrows():
        df.loc[i, 'Object'] = row['Object'] + '_Type' + str(row['Type']) + '_L' + str(row['Level'])
    
    # Retain necessary columns
    df = df.filter(['ID', 'Object', 'Volume'])

    # Filter and pivot by each ID to avoid duplicates
    df_wide = pd.DataFrame()
    # Get list of IDs to iterate over
    ids = self.get_id()
    for id in ids:
        df_id = df.loc[df['ID'] == id]
        # Drop second 'BasalForebrain_L/R_Type1_L5' row
        df_id = df_id.drop_duplicates(subset=['Object'], keep='first')
        df_id = df_id.pivot(index='ID', columns='Object', values='Volume')
        df_wide = pd.concat([df_wide, df_id])

    print(str(self.long_to_wide.__name__) + ": Conversion successful")
    return df_wide

def chat(self, key):
    llm = OpenAI(api_token=key)
    prompt = input(str(chat.__name__) + ': ')
    output = PandasAI(llm).run(self.df, prompt)
    
    return output

# Normalize covariate dataset by ICV, TBV, or ICV + TBV
def normalize_covariate_data(self, covariate_dataset, normalizing_factor: str):

    df = covariate_dataset.copy()
    normalized_cols = [col for col in df.columns if '_Type' in col]

    if normalizing_factor == 'icv_tbv':
        df['icv_tbv'] = df[['ICV', 'TBV']].sum(axis=1).to_frame()
        df[normalized_cols] = df[normalized_cols].div(df['icv_tbv'], axis=0)
        df = df.drop(columns=['icv_tbv'])

    if normalizing_factor == 'icv':
        df[normalized_cols] = df[normalized_cols].div(df['ICV'], axis=0)

    if normalizing_factor == 'tbv':
        df[normalized_cols] = df[normalized_cols].div(df['TBV'], axis=0)

    return df

if __name__ == '__main__':
    print(__name__)