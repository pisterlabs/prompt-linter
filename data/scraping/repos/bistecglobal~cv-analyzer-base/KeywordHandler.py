import openai
import spacy

def GetKeywords(inputText):
    
    prompt="Extract the keywords:"
    # Build the prompt by combining the provided prompt and input text
    full_prompt = f"{prompt} {inputText}"

    # Use OpenAI GPT-3 to generate keywords
    response = openai.Completion.create(
        engine="text-davinci-003",  # You can experiment with different engines
        prompt=full_prompt,
        max_tokens=100,  # Adjust max_tokens as needed
        n=1,  # Number of completions to generate
        stop=None,  # You can provide a list of strings to stop generation
        temperature=0.7  # Adjust temperature for randomness
    )

    # Extract keywords from the generated response
    keywords = response.choices[0].text.strip()
    
    return keywords

def ExtractKeywords(text):
    spacy.cli.download("en_core_web_sm")
    # Load the English language model
    nlp = spacy.load("en_core_web_sm")


    # Process the input text using spaCy
    doc = nlp(text)

    # Extract keywords (nouns and proper nouns)
    keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]

    return keywords