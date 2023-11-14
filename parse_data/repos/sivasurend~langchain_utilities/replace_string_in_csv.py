# a simple code to replace strings in columns of a csv

import pandas as pd

# load the csv file
df = pd.read_csv('file_path/example.csv')


def fill_template(row):
    template = row['Column1']
    product = row['Column2']
    template = template.replace('{string_to_be_replaced}', product)
    return template


# apply function to each row
df['Prompt'] = df.apply(fill_template, axis=1)

# write the result back to csv
df.to_csv('file_path/example.csv', index=False)
