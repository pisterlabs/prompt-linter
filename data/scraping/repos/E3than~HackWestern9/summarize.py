import cohere

def summarize_text(text):
    co = cohere.Client("XT6ArMCB57Gj7sOcfBDnK8Vwa8XC37bOvBokVF4I")

    response = co.generate( 
        model='xlarge', 
        prompt = text,
        max_tokens=100, 
        temperature=0.8,
        stop_sequences=["--"]
    )
    return response

