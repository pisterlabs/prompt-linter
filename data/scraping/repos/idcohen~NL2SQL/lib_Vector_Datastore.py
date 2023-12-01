import pandas as pd
 ## Embeddings
from openai.embeddings_utils import get_embedding

class VDS():
    def __init__(self, Filename):
        self._DBFilename_Load = Filename

    def Load_VDS_DF(self, Verbose=False):
        # for version 1, import dataframe
        try:
            df = pd.read_excel(self._DBFilename, sheet_name='VDS')
            if Verbose:
                print(f'Load_VDS_DF imported {df.shape[0]} rows from {self._DBFilename}')
            return 0
        except:
            print(f'Load_VDS_DF Error failed to import file {self._DBFilename}')
            return -1

        return df

    def Store_VDS_DF(self, df, Verbose=False):
        try:
            df.to_excel(self._DBFilename,sheet_name='VDS',index=False, header=True)
            if Verbose:
                print(f'Store_VDS_DF() wrote {df.shape[0]} rows to file {self._DBFilename}')
            return 0
        except:
            print(f'Store_VDS_DF Error failed to write to {self._DBFilename}')
            return -1
