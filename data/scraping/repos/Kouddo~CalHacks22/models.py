import cohere

bill = cohere.Client('redacted')
basic = cohere.Client('redacted')



def billModel(prompt):
    co = bill
    return co.generate( 
        model='xlarge', 
        prompt = prompt,
        max_tokens=40, 
        temperature=0.8,
        stop_sequences=["--"])

def basicmodel(prompt):
    co = basic
    return co.generate( 
        model='xlarge', 
        prompt = prompt,
        max_tokens=40, 
        temperature=0.8,
        stop_sequences=["--"])