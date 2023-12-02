import cohere

co = cohere.Client('rKNbSEzFUz1Naxs2ZwQQpZOL3IsPAY4pKLIpLDnG')

pprompt = "Extract the location from the following text: flower blossoms in tokyo japan"
response = co.generate(
            model='command-xlarge-nightly',
            
            prompt=pprompt,
            max_tokens=314,
            temperature=0.3,
            k=0,
            p=0.75,
            frequency_penalty=0,
            presence_penalty=0,
            stop_sequences=[],
            return_likelihoods='NONE'
        )

print('\nPrediction--- \n {}'.format(response.generations[0].text))