import os
# from pprint import pprint as pp

from importers.base_importer import BaseImporter
import pandas as pd
from openai import OpenAI

GC_TOKEN = os.getenv('GC_TOKEN')
ACCOUNT = os.getenv('GC_BANK1_ACCOUNT')
GC_IMPORTS_DIR = os.getenv('GC_BANK1_IMPORTS_DIR')

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class GoCardlessBankImporter(BaseImporter):

    def convert_to_df(self, data):
        for row in data:
            row['amount'] = row['transactionAmount']['amount']
            row['currency'] = row['transactionAmount']['currency']
            del row['transactionAmount']

        df = pd.DataFrame(data)
        self.logger.info(df)
        return df
    
    def parse_text(self, text):
        words = text.split()
        if len(words) <= 2:
            return pd.Series([' '.join(words), None])
        else:
            return pd.Series([' '.join(words[:2]), ' '.join(words[2:])])

    def split_account_and_desc(self, df):
        df[['Account', 'Description']] = df['entryReference'].apply(self.parse_text)
        return df
    
    def filter_month(self, df):
        df['date_obj'] = pd.to_datetime(df['bookingDate'])
        df = df[df['date_obj'].dt.month >= 10]
        df.drop(columns=['date_obj'], inplace=True)
        return df
    
    def run(self):
        self.empty_imports()
        data = self.get_data(account=ACCOUNT)
        df = self.convert_to_df(data)
        df = self.split_account_and_desc(df)
        df = self.filter_month(df)
        self.to_csv(df)
        self.copy_template()
        self.upload_to_firefly()

if __name__ == '__main__':
    gc = GoCardlessBankImporter(GC_IMPORTS_DIR)
    gc.run()