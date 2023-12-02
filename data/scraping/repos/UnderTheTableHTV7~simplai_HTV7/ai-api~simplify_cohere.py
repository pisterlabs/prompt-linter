import dotenv, os
import cohere
dotenv.load_dotenv('.env') # Use this in most cases
api_key = os.environ.get('COHERE_API_KEY')
co = cohere.Client(api_key)

def simplify_text_cohere(input_text):
    return co.generate(prompt=input_text, model="ed5b1a41-0e40-4071-979e-3f0204d119d4-ft", max_tokens=50, temperature=0.9).generations[0].text
