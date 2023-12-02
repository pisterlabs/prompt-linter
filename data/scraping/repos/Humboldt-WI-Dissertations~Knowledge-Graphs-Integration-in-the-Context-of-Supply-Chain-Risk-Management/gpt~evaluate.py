import pandas as pd

from openai_matcher import match_companies

def match_convert(entry):
    return [
        {
            'name': entry['Parsed_Company_SCM'],
            'lat': entry['lat_SCM'], 
            'long': entry['long_SCM'], 
        },
        {
            'name': entry['Parsed_Company_COYPU'],
            'lat': entry['lat_COYPU'], 
            'long': entry['long_COYPU'], 
        },
    ]

cn = pd.read_csv('cn.csv')
cn['country'] = cn.index.map(lambda x: 'cn')

de = pd.read_csv('de.csv')
de['country'] = de.index.map(lambda x: 'de')

us = pd.read_csv('us.csv')
us['country'] = us.index.map(lambda x: 'us')

eval_data = pd.concat([cn, de, us]) 

eval_data['match_GPT'] = eval_data.index.map(lambda x: '')
eval_data['match_confidence_GPT'] = eval_data.index.map(lambda x: '')

for index, row in eval_data.iterrows():
    companyA, companyB = match_convert(row)
    print(companyA, companyB)

    while True:
        try:
            res = match_companies(companyA, companyB)

            print(res)

            eval_data['match_GPT'][index] = res['match']
            eval_data['match_confidence_GPT'][index] = res['match_confidence']

            break
        except:
            print('Failed request retrying...')

    eval_data.to_csv('./evaluated.csv', index=False)
