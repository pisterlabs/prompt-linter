import cohere
import deepl
import json

COHERE_API = "Q8sALnI1zcKueF67Foy3aTx9NFCnk6B2sgR187dA"
DEEPL_API = "0fc72c72-d5dc-b28e-89c3-ca81522f9d64:fx"

def lambda_handler(event, context):


    fulltext = event['fulltext']
    
    if fulltext != "N/A" or None:
        
        co = cohere.Client(COHERE_API)

        engSum = co.summarize( 
            model='summarize-xlarge', 
            length='long',
            extractiveness='medium',
            temperature=0.5,
            text=event['fulltext']
        )

    
        translator = deepl.Translator(DEEPL_API)

        #result = translator.translate_text(engSum.summary, target_lang="JA")
        #jp_sum = result.text
        jp_sum = "placeholder value"
        en_sum = engSum.summary
    else:
        jp_sum = "Summary could not be translated"
        en_sum = "Summary could not be received"
    
    response = {
        'jpSum': jp_sum,
        'enSum': en_sum
    }
    
    return response